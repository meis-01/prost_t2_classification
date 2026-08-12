from prost_t2_classification.cli import build_parser


def test_train_cli_accepts_median_complex_pooling(tmp_path):
    args = build_parser().parse_args(
        [
            "train",
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--mode",
            "both",
            "--complex-pooling",
            "median",
        ]
    )

    assert args.complex_pooling == "median"


def test_train_cli_defaults_to_one_hundred_epochs(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert args.epochs == 100
