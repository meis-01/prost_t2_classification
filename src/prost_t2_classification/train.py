from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import nn
from tqdm import tqdm

from .dataset import make_dataloaders
from .logging_utils import get_logger, timestamp_slug
from .models import ComplexPooling, build_model


Mode = Literal["real", "complex", "complex_kspace"]


@dataclass(frozen=True)
class TrainConfig:
    manifest: Path
    runs_dir: Path
    mode: Mode
    epochs: int = 20
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = 10383
    num_workers: int = 0
    device: str | None = None
    complex_pooling: ComplexPooling = "max"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.mode not in ("real", "complex", "complex_kspace"):
            raise ValueError(f"Unknown model mode: {self.mode}")
        if self.complex_pooling not in ("max", "median", "average"):
            raise ValueError(f"Unknown complex pooling mode: {self.complex_pooling}")
        if self.mode == "complex_kspace" and self.complex_pooling != "average":
            raise ValueError("complex_kspace mode requires average complex pooling")


def train_both_models(
    manifest: Path,
    runs_dir: Path,
    **kwargs,
) -> dict[str, Path]:
    return {
        "real": train_model(
            TrainConfig(manifest=manifest, runs_dir=runs_dir, mode="real", **kwargs)
        ),
        "complex": train_model(
            TrainConfig(manifest=manifest, runs_dir=runs_dir, mode="complex", **kwargs)
        ),
    }


def train_model(config: TrainConfig) -> Path:
    logger = get_logger()
    set_seed(config.seed)
    run_label = run_label_from_config(config)
    run_dir = config.runs_dir / f"{timestamp_slug()}_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable_config = _serializable_config(config)
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
        seed=config.seed,
    )
    model = build_model(config.mode, complex_pooling=config.complex_pooling).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_score = -np.inf
    bad_epochs = 0
    history: list[dict[str, float]] = []
    history_path = run_dir / "history.csv"
    best_path = run_dir / f"best_{run_label}.pt"
    last_path = run_dir / f"last_{run_label}.pt"

    for epoch in range(config.epochs):
        train_stats = run_epoch(
            model,
            loaders.train,
            criterion,
            device=device,
            optimizer=optimizer,
            desc=f"{run_label} train {epoch + 1}/{config.epochs}",
            gradient_accumulation_steps=config.gradient_accumulation_steps,
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
        pd.DataFrame(history).to_csv(history_path, index=False)
        logger.info(
            "%s epoch=%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f",
            run_label,
            epoch + 1,
            train_stats["loss"],
            train_stats["auc"],
            val_stats["loss"],
            val_stats["auc"],
        )

        score = val_stats["auc"]
        if np.isnan(score):
            score = -val_stats["loss"]
        should_stop = False
        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    serializable_config,
                    epoch=epoch,
                    score=score,
                    best_score=best_score,
                    bad_epochs=bad_epochs,
                ),
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                should_stop = True
        torch.save(
            _checkpoint_payload(
                model,
                optimizer,
                serializable_config,
                epoch=epoch,
                score=score,
                best_score=best_score,
                bad_epochs=bad_epochs,
            ),
            last_path,
        )
        if should_stop:
            logger.info("Early stopping %s after %d bad validation epochs", run_label, bad_epochs)
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
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
    if config.mode == "complex_kspace":
        return "complex_kspace_modrelu_average_pool"
    if config.mode == "complex":
        if config.complex_pooling == "max":
            return "complex_modrelu"
        return f"complex_modrelu_{config.complex_pooling}_pool"
    return config.mode


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
    threshold: float = 0.5,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float]:
    loss, y_true, y_score = collect_epoch_outputs(
        model,
        loader,
        criterion,
        device=device,
        optimizer=optimizer,
        desc=desc,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return epoch_metrics(loss, y_true, y_score, threshold=threshold)


def collect_epoch_outputs(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    accumulated_samples = 0
    accumulated_batches = 0
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(is_train):
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device).float().flatten()
            batch_samples = targets.numel()
            logits = model(inputs).flatten()
            loss = criterion(logits, targets)
            if is_train:
                (loss * batch_samples).backward()
                accumulated_samples += batch_samples
                accumulated_batches += 1
                if accumulated_batches == gradient_accumulation_steps:
                    _step_optimizer(optimizer, model, accumulated_samples)
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_samples = 0
                    accumulated_batches = 0
            total_loss += float(loss.detach().cpu().item()) * batch_samples
            total_samples += batch_samples
            all_targets.append(targets.detach().cpu().numpy())
            all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

    if is_train and accumulated_samples:
        _step_optimizer(optimizer, model, accumulated_samples)
        optimizer.zero_grad(set_to_none=True)
    if total_samples == 0:
        raise ValueError("Cannot run an epoch with an empty data loader.")
    y_true = np.concatenate(all_targets).astype(np.int32)
    y_score = np.concatenate(all_scores).astype(np.float32)
    return total_loss / total_samples, y_true, y_score


def _step_optimizer(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    accumulated_samples: int,
) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(accumulated_samples)
    optimizer.step()


def epoch_metrics(
    loss: float,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
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


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
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


def _serializable_config(config: TrainConfig) -> dict[str, object]:
    data = asdict(config)
    data["manifest"] = str(config.manifest)
    data["runs_dir"] = str(config.runs_dir)
    return data


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
    *,
    epoch: int,
    score: float,
    best_score: float,
    bad_epochs: int,
) -> dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "score": float(score),
        "best_score": float(best_score),
        "bad_epochs": int(bad_epochs),
    }
