from __future__ import annotations

import argparse
import gc
import os

import psutil
import torch

from prost_t2_classification.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    process = psutil.Process(os.getpid())
    available_gb = psutil.virtual_memory().available / 1024**3
    print(f"available_ram_gb={available_gb:.2f}")

    parameter_counts = {
        mode: sum(parameter.numel() for parameter in build_model(mode, in_channels=args.channels).parameters())
        for mode in ("real", "complex")
    }
    ratio = parameter_counts["real"] / parameter_counts["complex"]
    print(f"parameters={parameter_counts} real_to_complex_ratio={ratio:.4f}")

    for mode in ("real", "complex"):
        model = build_model(mode, in_channels=args.channels)
        real = torch.randn(args.batch_size, args.channels, args.image_size, args.image_size)
        inputs = torch.complex(real, torch.randn_like(real)) if mode == "complex" else real
        before = process.memory_info().rss
        output = model(inputs)
        after_forward = process.memory_info().rss
        output.mean().backward()
        after_backward = process.memory_info().rss
        print(
            f"{mode}: before_gb={before / 1024**3:.3f} "
            f"forward_gb={after_forward / 1024**3:.3f} "
            f"backward_gb={after_backward / 1024**3:.3f}"
        )
        del model, real, inputs, output
        gc.collect()


if __name__ == "__main__":
    main()
