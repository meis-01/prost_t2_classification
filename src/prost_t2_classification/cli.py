from __future__ import annotations

import argparse
from pathlib import Path

from .logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prost-t2",
        description="Train the paired real and complex T2 classifiers from prepared four-coil NPZ data.",
    )
    parser.add_argument("--log-dir", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train real, complex, or both models.")
    train_parser.add_argument("--manifest", type=Path, required=True)
    train_parser.add_argument("--runs-dir", type=Path, required=True)
    train_parser.add_argument("--mode", choices=("real", "complex", "both"), default="both")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=10383)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument(
        "--complex-pooling",
        choices=("max", "median", "average"),
        default="max",
        help="Intermediate pooling used by the complex model.",
    )
    train_parser.add_argument(
        "--complex-input-domain", choices=("image", "kspace"), default="image"
    )
    train_parser.add_argument(
        "--complex-normalization", choices=("rms", "batchnorm"), default="rms"
    )
    train_parser.add_argument(
        "--complex-convolution", choices=("standard", "widely_linear"), default="standard"
    )
    train_parser.add_argument(
        "--complex-streams", choices=("complex_only", "dual"), default="complex_only"
    )
    train_parser.add_argument(
        "--complex-interaction",
        choices=("none", "modulus_gate", "holographic"),
        default="none",
    )
    train_parser.add_argument(
        "--complex-activation",
        choices=("modrelu", "magnitude_silu", "crelu", "cardioid"),
        default="modrelu",
    )
    train_parser.set_defaults(func=cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_dir)
    try:
        return int(args.func(args))
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")


def cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train_both_models, train_model

    common = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "device": args.device,
        "complex_pooling": args.complex_pooling,
        "complex_input_domain": args.complex_input_domain,
        "complex_normalization": args.complex_normalization,
        "complex_convolution": args.complex_convolution,
        "complex_streams": args.complex_streams,
        "complex_interaction": args.complex_interaction,
        "complex_activation": args.complex_activation,
    }
    if args.mode == "both":
        train_both_models(args.manifest, args.runs_dir, **common)
    else:
        train_model(
            TrainConfig(
                manifest=args.manifest,
                runs_dir=args.runs_dir,
                mode=args.mode,
                **common,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
