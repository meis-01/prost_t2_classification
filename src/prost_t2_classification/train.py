from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import nn
from tqdm import tqdm

from .dataset import make_dataloaders
from .logging_utils import get_logger, timestamp_slug
from .models import build_model


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
    in_channels: int = 5
    dropout: float = 0.2
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
        if self.in_channels < 1:
            raise ValueError("in_channels must be at least 1.")


def train_both_models(
    manifest: Path,
    runs_dir: Path,
    **kwargs,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    for mode in ("real", "complex"):
        config = TrainConfig(manifest=manifest, runs_dir=runs_dir, mode=mode, **kwargs)
        outputs[mode] = train_model(config)
    return outputs


def train_model(config: TrainConfig) -> Path:
    logger = get_logger()
    set_seed(config.seed)
    run_dir = config.runs_dir / f"{timestamp_slug()}_{config.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(_serializable_config(config), indent=2),
        encoding="utf-8",
    )

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Training %s model on %s", config.mode, device)

    loaders = make_dataloaders(
        config.manifest,
        mode=config.mode,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    model = build_model(config.mode, in_channels=config.in_channels, dropout=config.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([loaders.pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_score = -np.inf
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    best_path = run_dir / f"best_{config.mode}.pt"

    for epoch in range(config.epochs):
        train_stats = run_epoch(
            model,
            loaders.train,
            criterion,
            device=device,
            optimizer=optimizer,
            desc=f"{config.mode} train {epoch + 1}/{config.epochs}",
        )
        val_stats = run_epoch(
            model,
            loaders.validation,
            criterion,
            device=device,
            optimizer=None,
            desc=f"{config.mode} val {epoch + 1}/{config.epochs}",
        )
        record = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items()},
        }
        history.append(record)
        logger.info(
            "%s epoch=%d train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f",
            config.mode,
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
            torch.save({"model_state": model.state_dict(), "config": _serializable_config(config)}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                logger.info("Early stopping %s after %d bad validation epochs", config.mode, bad_epochs)
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_stats = run_epoch(
        model,
        loaders.test,
        criterion,
        device=device,
        optimizer=None,
        desc=f"{config.mode} test",
    )
    (run_dir / "test_metrics.json").write_text(json.dumps(test_stats, indent=2), encoding="utf-8")
    logger.info("%s test metrics: %s", config.mode, test_stats)
    return run_dir


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    desc: str,
) -> Dict[str, float]:
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
    return {"loss": float(np.mean(losses)), **binary_metrics(y_true, y_score)}


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_pred = (y_score >= 0.5).astype(np.int32)
    out = {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(metrics.roc_auc_score(y_true, y_score))
        out["average_precision"] = float(metrics.average_precision_score(y_true, y_score))
    else:
        out["auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out


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
