from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .download import download_entries, extract_archives, parse_prostate_download_script
from .logging_utils import configure_logging


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_dir = Path(args.log_dir) if getattr(args, "log_dir", None) else None
    configure_logging(log_dir)
    try:
        return args.func(args)
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prost-t2")
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional directory for pipeline.log")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download and extract labels plus T2 archives")
    download_parser.add_argument("--download-script", type=Path, default=Path("prostate_download_script.txt"))
    download_parser.add_argument("--download-dir", type=Path, required=True)
    download_parser.add_argument("--extract-dir", type=Path, default=None)
    download_parser.add_argument("--overwrite", action="store_true")
    download_parser.add_argument("--no-extract", action="store_true")
    download_parser.add_argument("--dry-run", action="store_true")
    download_parser.set_defaults(func=cmd_download)

    recon_parser = subparsers.add_parser("reconstruct", help="Run fastmri-tools reconstruction for T2 H5 files")
    recon_parser.add_argument("--raw-root", type=Path, required=True)
    recon_parser.add_argument("--recon-dir", type=Path, required=True)
    recon_parser.add_argument("--kernel-size", default="5,5")
    recon_parser.add_argument("--overwrite", action="store_true")
    recon_parser.add_argument("--limit", type=int, default=None)
    recon_parser.set_defaults(func=cmd_reconstruct)

    npz_parser = subparsers.add_parser("make-npz", help="Create selected-coil NPZ files and manifest")
    npz_parser.add_argument("--labels", type=Path, required=True, help="Label CSV or directory containing it")
    npz_parser.add_argument("--recon-dir", type=Path, required=True)
    npz_parser.add_argument("--npz-dir", type=Path, required=True)
    npz_parser.add_argument("--crop-size", type=int, default=224)
    npz_parser.add_argument("--max-coils", type=int, default=5)
    npz_parser.add_argument("--overwrite", action="store_true")
    npz_parser.add_argument("--limit-patients", type=int, default=None)
    npz_parser.add_argument("--limit-slices", type=int, default=None)
    npz_parser.set_defaults(func=cmd_make_npz)

    prepare_parser = subparsers.add_parser(
        "prepare-npz",
        help="Run T2 reconstruction if needed, then create selected-coil NPZ files",
    )
    prepare_parser.add_argument("--raw-root", type=Path, required=True)
    prepare_parser.add_argument("--labels", type=Path, default=None, help="Label CSV or directory; defaults to raw root")
    prepare_parser.add_argument("--recon-dir", type=Path, required=True)
    prepare_parser.add_argument("--npz-dir", type=Path, required=True)
    prepare_parser.add_argument("--crop-size", type=int, default=224)
    prepare_parser.add_argument("--max-coils", type=int, default=5)
    prepare_parser.add_argument("--kernel-size", default="5,5")
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument("--skip-reconstruct", action="store_true")
    prepare_parser.add_argument("--limit", type=int, default=None, help="Limit reconstructed files for smoke tests")
    prepare_parser.add_argument("--limit-patients", type=int, default=None)
    prepare_parser.add_argument("--limit-slices", type=int, default=None)
    prepare_parser.set_defaults(func=cmd_prepare_npz)

    train_parser = subparsers.add_parser("train", help="Train real, complex, or both classifiers")
    add_train_args(train_parser)
    train_parser.set_defaults(func=cmd_train)

    run_parser = subparsers.add_parser("run", help="Run any contiguous part of the full pipeline")
    run_parser.add_argument("--download-script", type=Path, default=Path("prostate_download_script.txt"))
    run_parser.add_argument("--download-dir", type=Path, default=None)
    run_parser.add_argument("--extract-dir", type=Path, default=None)
    run_parser.add_argument("--recon-dir", type=Path, default=None)
    run_parser.add_argument("--npz-dir", type=Path, default=None)
    run_parser.add_argument("--runs-dir", type=Path, default=None)
    run_parser.add_argument("--skip-download", action="store_true")
    run_parser.add_argument("--skip-extract", action="store_true")
    run_parser.add_argument("--skip-reconstruct", action="store_true")
    run_parser.add_argument("--skip-npz", action="store_true")
    run_parser.add_argument("--skip-train", action="store_true")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--crop-size", type=int, default=224)
    run_parser.add_argument("--max-coils", type=int, default=5)
    run_parser.add_argument("--kernel-size", default="5,5")
    add_train_args(run_parser, include_manifest=False, include_runs_dir=False)
    run_parser.set_defaults(func=cmd_run)
    return parser


def add_train_args(
    parser: argparse.ArgumentParser,
    *,
    include_manifest: bool = True,
    include_runs_dir: bool = True,
) -> None:
    if include_manifest:
        parser.add_argument("--manifest", type=Path, required=True)
    if include_runs_dir:
        parser.add_argument("--runs-dir", type=Path, required=include_manifest)
    parser.add_argument("--mode", choices=("real", "complex", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10383)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)


def cmd_download(args) -> int:
    if not args.no_extract and args.extract_dir is None:
        raise ValueError("--extract-dir is required unless --no-extract is set.")
    entries = parse_prostate_download_script(args.download_script)
    paths = download_entries(entries, args.download_dir, overwrite=args.overwrite, dry_run=args.dry_run)
    if not args.no_extract:
        extract_archives(paths, args.extract_dir, overwrite=args.overwrite, dry_run=args.dry_run)
    return 0


def cmd_reconstruct(args) -> int:
    from .preprocess import reconstruct_t2_dataset

    kernel_size = parse_kernel_size(args.kernel_size)
    reconstruct_t2_dataset(
        args.raw_root,
        args.recon_dir,
        kernel_size=kernel_size,
        skip_existing=not args.overwrite,
        limit=args.limit,
    )
    return 0


def cmd_make_npz(args) -> int:
    from .preprocess import make_npz_dataset

    make_npz_dataset(
        args.labels,
        args.recon_dir,
        args.npz_dir,
        crop_size=args.crop_size,
        max_coils=args.max_coils,
        overwrite=args.overwrite,
        limit_patients=args.limit_patients,
        limit_slices=args.limit_slices,
    )
    return 0


def cmd_prepare_npz(args) -> int:
    from .preprocess import make_npz_dataset, reconstruct_t2_dataset

    if not args.skip_reconstruct:
        reconstruct_t2_dataset(
            args.raw_root,
            args.recon_dir,
            kernel_size=parse_kernel_size(args.kernel_size),
            skip_existing=not args.overwrite,
            limit=args.limit,
        )
    make_npz_dataset(
        args.labels or args.raw_root,
        args.recon_dir,
        args.npz_dir,
        crop_size=args.crop_size,
        max_coils=args.max_coils,
        overwrite=args.overwrite,
        limit_patients=args.limit_patients,
        limit_slices=args.limit_slices,
    )
    return 0


def cmd_train(args) -> int:
    train_from_args(args.manifest, args)
    return 0


def cmd_run(args) -> int:
    base = Path.cwd()
    needs_download_dir = not args.skip_download
    needs_extract_dir = not args.skip_download or not args.skip_reconstruct or not args.skip_npz
    needs_recon_dir = not args.skip_reconstruct or not args.skip_npz
    needs_npz_dir = not args.skip_npz or not args.skip_train
    needs_runs_dir = not args.skip_train

    download_dir = (
        choose_path(args.download_dir, "Archive download directory", base / "data" / "archives")
        if needs_download_dir
        else args.download_dir
    )
    extract_dir = (
        choose_path(args.extract_dir, "Extracted raw data directory", base / "data" / "raw")
        if needs_extract_dir
        else args.extract_dir
    )
    recon_dir = (
        choose_path(args.recon_dir, "Reconstruction output directory", base / "data" / "recon_t2")
        if needs_recon_dir
        else args.recon_dir
    )
    npz_dir = (
        choose_path(args.npz_dir, "Selected-coil NPZ directory", base / "data" / "npz_t2_coils")
        if needs_npz_dir
        else args.npz_dir
    )
    runs_dir = (
        choose_path(args.runs_dir, "Training runs directory", base / "runs")
        if needs_runs_dir
        else args.runs_dir
    )

    if not args.skip_download:
        entries = parse_prostate_download_script(args.download_script)
        archives = download_entries(entries, download_dir, overwrite=args.overwrite)
        if not args.skip_extract:
            extract_archives(archives, extract_dir, overwrite=args.overwrite)

    if not args.skip_reconstruct:
        from .preprocess import reconstruct_t2_dataset

        reconstruct_t2_dataset(
            extract_dir,
            recon_dir,
            kernel_size=parse_kernel_size(args.kernel_size),
            skip_existing=not args.overwrite,
        )

    manifest = npz_dir / "manifest.csv" if npz_dir is not None else None
    if not args.skip_npz:
        from .preprocess import make_npz_dataset

        manifest = make_npz_dataset(
            extract_dir,
            recon_dir,
            npz_dir,
            crop_size=args.crop_size,
            max_coils=args.max_coils,
            overwrite=args.overwrite,
        )

    if not args.skip_train:
        if manifest is None:
            raise ValueError("Training requires an NPZ manifest; provide --npz-dir or run make-npz first.")
        args.runs_dir = runs_dir
        train_from_args(manifest, args)
    return 0


def train_from_args(manifest: Path, args) -> None:
    from .train import TrainConfig, train_both_models, train_model

    common = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "device": args.device,
    }
    if args.mode == "both":
        train_both_models(manifest, args.runs_dir, **common)
        return
    train_model(TrainConfig(manifest=manifest, runs_dir=args.runs_dir, mode=args.mode, **common))


def choose_path(value: Optional[Path], label: str, default: Path) -> Path:
    if value is not None:
        return value
    if sys.stdin.isatty():
        entered = input(f"{label} [{default}]: ").strip()
        return Path(entered) if entered else default
    return default


def parse_kernel_size(value: str) -> tuple[int, int]:
    try:
        first, second = value.split(",", 1)
        kernel = (int(first), int(second))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("kernel size must look like '5,5'.") from exc
    if kernel[0] <= 0 or kernel[1] <= 0 or kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
        raise argparse.ArgumentTypeError("kernel size values must be positive odd integers.")
    return kernel


if __name__ == "__main__":
    raise SystemExit(main())
