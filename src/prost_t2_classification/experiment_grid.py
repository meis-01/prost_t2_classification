from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path


INPUT_DOMAINS = ("image", "kspace")
POOLINGS = ("max", "median", "average")
NORMALIZATIONS = ("rms", "batchnorm")
CONVOLUTIONS = ("standard", "widely_linear")
ACTIVATIONS = ("modrelu", "magnitude_silu", "crelu", "cardioid")
STREAM_INTERACTIONS = (
    ("complex_only", "none"),
    ("complex_only", "holographic"),
    ("dual", "none"),
    ("dual", "modulus_gate"),
    ("dual", "holographic"),
)


@dataclass(frozen=True)
class ExperimentModelSpec:
    model_key: str
    mode: str
    input_domain: str
    pooling: str
    normalization: str
    convolution: str
    streams: str
    interaction: str
    activation: str

    @property
    def is_real(self) -> bool:
        return self.mode == "real"


def _complex_key(
    input_domain: str,
    pooling: str,
    normalization: str,
    convolution: str,
    streams: str,
    interaction: str,
    activation: str,
) -> str:
    canonical = (
        input_domain == "image"
        and normalization == "rms"
        and convolution == "standard"
        and streams == "complex_only"
        and interaction == "none"
        and activation == "modrelu"
    )
    if canonical:
        return f"complex_{pooling}"

    aliases = {
        "image": "img",
        "kspace": "ksp",
        "median": "med",
        "average": "avg",
        "batchnorm": "cbn",
        "standard": "std",
        "widely_linear": "wl",
        "complex_only": "single",
        "modulus_gate": "gate",
        "holographic": "holo",
        "magnitude_silu": "magsilu",
        "cardioid": "card",
    }
    values = (
        input_domain,
        pooling,
        normalization,
        convolution,
        streams,
        interaction,
        activation,
    )
    return "cx_" + "_".join(aliases.get(value, value) for value in values)


def model_key_for_complex(
    *,
    input_domain: str,
    pooling: str,
    normalization: str,
    convolution: str,
    streams: str,
    interaction: str,
    activation: str,
) -> str:
    return _complex_key(
        input_domain,
        pooling,
        normalization,
        convolution,
        streams,
        interaction,
        activation,
    )


def _complex_spec(values: tuple[str, str, str, str, tuple[str, str], str]) -> ExperimentModelSpec:
    input_domain, pooling, normalization, convolution, stream_interaction, activation = values
    streams, interaction = stream_interaction
    return ExperimentModelSpec(
        model_key=_complex_key(
            input_domain,
            pooling,
            normalization,
            convolution,
            streams,
            interaction,
            activation,
        ),
        mode="complex",
        input_domain=input_domain,
        pooling=pooling,
        normalization=normalization,
        convolution=convolution,
        streams=streams,
        interaction=interaction,
        activation=activation,
    )


def experiment_model_specs() -> tuple[ExperimentModelSpec, ...]:
    real = ExperimentModelSpec(
        model_key="real",
        mode="real",
        input_domain="image",
        pooling="none",
        normalization="batchnorm",
        convolution="real",
        streams="magnitude_only",
        interaction="none",
        activation="silu",
    )
    values = tuple(
        product(
            INPUT_DOMAINS,
            POOLINGS,
            NORMALIZATIONS,
            CONVOLUTIONS,
            STREAM_INTERACTIONS,
            ACTIVATIONS,
        )
    )
    complex_specs = tuple(_complex_spec(value) for value in values)
    canonical_keys = ("complex_max", "complex_median", "complex_average")
    by_key = {spec.model_key: spec for spec in complex_specs}
    ordered = tuple(by_key[key] for key in canonical_keys) + tuple(
        spec for spec in complex_specs if spec.model_key not in canonical_keys
    )
    specs = (real, *ordered)
    keys = [spec.model_key for spec in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Experiment grid generated duplicate model keys.")
    if len(specs) != 481:
        raise RuntimeError(f"Expected 481 grid models including real; found {len(specs)}.")
    return specs


def model_spec_from_index(index: int) -> ExperimentModelSpec:
    specs = experiment_model_specs()
    if not 0 <= index < len(specs):
        raise IndexError(f"Model index must be in [0, {len(specs) - 1}]; got {index}.")
    return specs[index]


def model_spec_from_key(model_key: str) -> ExperimentModelSpec:
    for spec in experiment_model_specs():
        if spec.model_key == model_key:
            return spec
    raise KeyError(f"Unknown experiment model key: {model_key}")


def write_grid(csv_path: Path, json_path: Path) -> None:
    records = [asdict(spec) for spec in experiment_model_specs()]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or write the fixed complex-model experiment grid.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("count")
    row = subparsers.add_parser("row")
    row.add_argument("--index", required=True, type=int)
    write = subparsers.add_parser("write")
    write.add_argument("--csv", required=True, type=Path)
    write.add_argument("--json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "count":
        print(len(experiment_model_specs()))
    elif args.command == "row":
        spec = model_spec_from_index(args.index)
        print("\t".join(str(value) for value in asdict(spec).values()))
    else:
        write_grid(args.csv, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
