from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import nn
from tqdm import tqdm

from .dataset import make_dataloaders
from .logging_utils import get_logger, timestamp_slug
from .models import (
    COMPLEX_ACTIVATIONS,
    PARAMETER_MATCHED_REAL_CHANNELS,
    ComplexActivation,
    build_model,
)


Mode = Literal["real", "complex"]


@dataclass(frozen=True)
class TrainConfig:
    manifest: Path
    runs_dir: Path
    mode: Mode
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = 10383
    num_workers: int = 0
    in_channels: Optional[int] = None
    dropout: float = 0.2
    real_channels: Tuple[int, int, int, int] = PARAMETER_MATCHED_REAL_CHANNELS
    complex_activation: ComplexActivation = "modrelu"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")
        if self.in_channels is not None and self.in_channels < 1:
            raise ValueError("in_channels must be at least 1.")
        if len(self.real_channels) != 4 or any(channel < 1 for channel in self.real_channels):
            raise ValueError("real_channels must contain four positive channel counts.")
        if self.complex_activation not in COMPLEX_ACTIVATIONS:
            raise ValueError(
                f"Unknown complex activation {self.complex_activation!r}; "
                f"expected one of {', '.join(COMPLEX_ACTIVATIONS)}."
            )


def train_both_models(
    manifest: Path,
    runs_dir: Path,
    *,
    complex_activations: tuple[ComplexActivation, ...] = ("modrelu",),
    **kwargs,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    outputs["real"] = train_model(TrainConfig(manifest=manifest, runs_dir=runs_dir, mode="real", **kwargs))
    for activation in complex_activations:
        config = TrainConfig(
            manifest=manifest,
            runs_dir=runs_dir,
            mode="complex",
            complex_activation=activation,
            **kwargs,
        )
        key = "complex" if len(complex_activations) == 1 else f"complex_{activation}"
        outputs[key] = train_model(config)
    return outputs


def train_model(config: TrainConfig) -> Path:
    logger = get_logger()
    set_seed(config.seed)
    run_label = run_label_from_config(config)
    run_dir = config.runs_dir / f"{timestamp_slug()}_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    in_channels = resolve_in_channels(config)
    serializable_config = _serializable_config(config)
    serializable_config["resolved_in_channels"] = in_channels
    (run_dir / "config.json").write_text(
        json.dumps(serializable_config, indent=2),
        encoding="utf-8",
    )

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Training %s model on %s", run_label, device)

    loaders = make_dataloaders(
        config.manifest,
        mode=config.mode,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    model = build_model(
        config.mode,
        in_channels=in_channels,
        dropout=config.dropout,
        real_channels=config.real_channels,
        complex_activation=config.complex_activation,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([loaders.pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_score = -np.inf
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    best_path = run_dir / f"best_{run_label}.pt"

    for epoch in range(config.epochs):
        train_stats = run_epoch(
            model,
            loaders.train,
            criterion,
            device=device,
            optimizer=optimizer,
            desc=f"{run_label} train {epoch + 1}/{config.epochs}",
        )
        val_stats = run_epoch(
            model,
            loaders.validation,
            criterion,
            device=device,
            optimizer=None,
            desc=f"{run_label} val {epoch + 1}/{config.epochs}",
        )
        record = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items()},
        }
        history.append(record)
        logger.info(
            "%s epoch=%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f",
            run_label,
            epoch,
            train_stats["loss"],
            train_stats["auc"],
            val_stats["loss"],
            val_stats["auc"],
        )

        score = val_stats["auc"]
        if np.isnan(score):
            score = -val_stats["loss"]
        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": serializable_config}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                logger.info("Early stopping %s after %d bad validation epochs", run_label, bad_epochs)
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_loss, val_true, val_score = collect_epoch_outputs(
        model,
        loaders.validation,
        criterion,
        device=device,
        optimizer=None,
        desc=f"{run_label} tune threshold",
    )
    threshold = tune_threshold(val_true, val_score)
    val_tuned_stats = epoch_metrics(val_loss, val_true, val_score, threshold=threshold)
    threshold_stats = {
        "metric": "balanced_accuracy",
        "threshold": float(threshold),
        "validation_at_threshold": val_tuned_stats,
        "validation_at_0_5": epoch_metrics(val_loss, val_true, val_score, threshold=0.5),
    }
    (run_dir / "threshold.json").write_text(json.dumps(threshold_stats, indent=2), encoding="utf-8")
    logger.info(
        "%s tuned threshold=%.4f val_balanced_accuracy=%.4f",
        run_label,
        threshold,
        val_tuned_stats["balanced_accuracy"],
    )

    test_stats = run_epoch(
        model,
        loaders.test,
        criterion,
        device=device,
        optimizer=None,
        desc=f"{run_label} test",
        threshold=threshold,
    )
    (run_dir / "test_metrics.json").write_text(json.dumps(test_stats, indent=2), encoding="utf-8")
    logger.info("%s test metrics: %s", run_label, test_stats)
    return run_dir


def run_label_from_config(config: TrainConfig) -> str:
    if config.mode == "complex":
        return f"complex_{config.complex_activation}"
    return config.mode


def resolve_in_channels(config: TrainConfig) -> int:
    manifest_channels = infer_manifest_channels(config.manifest)
    if config.in_channels is None:
        return manifest_channels
    if config.in_channels != manifest_channels:
        raise ValueError(
            f"--in-channels is {config.in_channels}, but manifest samples contain {manifest_channels} channel(s)."
        )
    return config.in_channels


def infer_manifest_channels(manifest_path: Path) -> int:
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"{manifest_path} has no samples.")
    if "channels" in manifest.columns:
        channels = sorted({int(value) for value in manifest["channels"].dropna().tolist()})
        if len(channels) == 1 and channels[0] > 0:
            return channels[0]
        raise ValueError(f"{manifest_path} must contain one positive channel count; found {channels}.")

    sample_path = manifest_path.parent / str(manifest.iloc[0]["path"])
    with np.load(sample_path) as npz:
        image_complex = npz["image_complex"]
        if image_complex.ndim != 3 or image_complex.shape[0] <= 0:
            raise ValueError(f"{sample_path} image_complex must have shape (channels, height, width).")
        return int(image_complex.shape[0])


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    desc: str,
    threshold: float = 0.5,
) -> Dict[str, float]:
    loss, y_true, y_score = collect_epoch_outputs(
        model,
        loader,
        criterion,
        device=device,
        optimizer=optimizer,
        desc=desc,
    )
    return epoch_metrics(loss, y_true, y_score, threshold=threshold)


def collect_epoch_outputs(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    desc: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: List[float] = []
    all_targets: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    with torch.set_grad_enabled(is_train):
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device).float().flatten()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs).flatten()
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            all_targets.append(targets.detach().cpu().numpy())
            all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

    y_true = np.concatenate(all_targets).astype(np.int32)
    y_score = np.concatenate(all_scores).astype(np.float32)
    return float(np.mean(losses)), y_true, y_score


def epoch_metrics(
    loss: float,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    return {"loss": float(loss), **binary_metrics(y_true, y_score, threshold=threshold)}


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)
    if len(np.unique(y_true)) < 2:
        return 0.5

    unique_scores = np.unique(y_score)
    if len(unique_scores) > 1:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates = np.concatenate(([0.0, 0.5, 1.0], unique_scores, midpoints))
    else:
        candidates = np.asarray([0.0, 0.5, 1.0, unique_scores[0]], dtype=np.float32)

    best_threshold = 0.5
    best_score = -np.inf
    best_distance = np.inf
    for threshold in np.unique(candidates):
        balanced_accuracy = binary_metrics(y_true, y_score, threshold=float(threshold))["balanced_accuracy"]
        if np.isnan(balanced_accuracy):
            continue
        distance = abs(float(threshold) - 0.5)
        if balanced_accuracy > best_score + 1e-12 or (
            abs(balanced_accuracy - best_score) <= 1e-12 and distance < best_distance
        ):
            best_score = balanced_accuracy
            best_threshold = float(threshold)
            best_distance = distance
    return best_threshold


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int32)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    if np.isnan(sensitivity) or np.isnan(specificity):
        balanced_accuracy = float("nan")
    else:
        balanced_accuracy = float((sensitivity + specificity) / 2.0)

    out = {
        "threshold": float(threshold),
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(metrics.roc_auc_score(y_true, y_score))
        out["average_precision"] = float(metrics.average_precision_score(y_true, y_score))
    else:
        out["auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _serializable_config(config: TrainConfig) -> Dict[str, object]:
    data = asdict(config)
    data["manifest"] = str(config.manifest)
    data["runs_dir"] = str(config.runs_dir)
    return data
