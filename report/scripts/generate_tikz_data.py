"""Export CSV-derived data files consumed by native PGFPlots figures."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "derived"
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

def write(path, frame, columns):
    frame.loc[:, columns].to_csv(path, sep=" ", index=False, header=True, float_format="%.8f")

def main():
    d = pd.read_csv(DERIVED / "phase1_annotated.csv")
    cx = d[d["mode"] == "complex"].copy()
    real = d.loc[d.run_label == "real"].iloc[0]
    rng = np.random.default_rng(10383)
    image = cx[cx.input_domain == "image"].copy(); image["x"] = rng.uniform(-0.085, 0.085, len(image)); image["auc"] = image.test_auc
    fourier = cx[cx.input_domain == "kspace"].copy(); fourier["x"] = 1 + rng.uniform(-0.085, 0.085, len(fourier)); fourier["auc"] = fourier.test_auc
    write(DATA / "landscape_image.dat", image, ["x", "auc"]); write(DATA / "landscape_fourier.dat", fourier, ["x", "auc"])
    pd.DataFrame({"x": [-0.15, 1.15], "auc": [real.test_auc, real.test_auc]}).to_csv(DATA / "real_line.dat", sep=" ", index=False, float_format="%.8f")
    landscape_summary = cx.groupby("input_domain").test_auc.agg(
        mean="mean", median="median",
        q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75)
    ).reindex(["image", "kspace"]).reset_index()
    landscape_summary["x"] = [0, 1]
    landscape_summary.to_csv(DATA / "landscape_summary.dat", sep=" ", index=False, float_format="%.8f")
    landscape_summary[landscape_summary.input_domain == "image"].to_csv(
        DATA / "landscape_summary_image.dat", sep=" ", index=False, float_format="%.8f"
    )
    landscape_summary[landscape_summary.input_domain == "kspace"].to_csv(
        DATA / "landscape_summary_fourier.dat", sep=" ", index=False, float_format="%.8f"
    )

    e = pd.read_csv(DERIVED / "main_effects.csv")
    desired = [("normalization", "rms"), ("convolution", "widely_linear"), ("pooling", "max"), ("activation", "magnitude_silu"), ("activation", "cardioid"), ("pooling", "median"), ("activation", "crelu"), ("convolution", "standard"), ("pooling", "average"), ("activation", "modrelu"), ("normalization", "batchnorm")]
    wanted = pd.DataFrame([e[(e.factor == factor) & (e.level == level)].iloc[0] for factor, level in desired])
    wanted["idx"] = np.arange(len(wanted))
    write(DATA / "marginal_effects.dat", wanted.rename(columns={"effect_auc":"effect"}), ["idx", "effect"])
    norm = cx.groupby(["normalization", "input_domain"]).test_auc.mean().unstack().reset_index()
    norm["x"] = np.arange(len(norm)); write(DATA / "domain_normalization.dat", norm.rename(columns={"image":"image_auc", "kspace":"fourier_auc"}), ["x", "image_auc", "fourier_auc"])
    order = [("complex_only", "none"), ("complex_only", "holographic"), ("dual", "none"), ("dual", "modulus_gate"), ("dual", "holographic")]
    stream = cx.groupby(["input_domain", "streams", "interaction"]).test_auc.mean().reset_index()
    rows=[]
    for x,(streams,interaction) in enumerate(order):
        row={"x":x}
        for domain,key in [("image","image_auc"),("kspace","fourier_auc")]: row[key]=stream[(stream.input_domain==domain)&(stream.streams==streams)&(stream.interaction==interaction)].test_auc.iloc[0]
        rows.append(row)
    pd.DataFrame(rows).to_csv(DATA / "domain_stream.dat", sep=" ", index=False, float_format="%.8f")

    canonical = d[d.run_label.isin(["real","complex_modrelu","complex_modrelu_median_pool","complex_modrelu_average_pool"])].copy()
    canonical["x"] = range(len(canonical)); canonical["name"] = ["Real", "Max", "Median", "Average"]
    write(DATA / "canonical.dat", canonical.rename(columns={"test_auc":"auc", "test_average_precision":"ap"}), ["x", "auc", "ap"])

    image["auc"] = image.test_auc; image["ap"] = image.test_average_precision
    fourier["auc"] = fourier.test_auc; fourier["ap"] = fourier.test_average_precision
    write(DATA / "scatter_image.dat", image, ["auc", "ap"]); write(DATA / "scatter_fourier.dat", fourier, ["auc", "ap"])
    can = canonical.copy(); can["auc"] = can.test_auc; can["ap"] = can.test_average_precision
    write(DATA / "scatter_canonical.dat", can, ["auc", "ap"])
    best = cx.loc[cx.test_auc.idxmax()]
    pd.DataFrame({"auc":[best.test_auc],"ap":[best.test_average_precision]}).to_csv(DATA / "scatter_best.dat", sep=" ", index=False, float_format="%.8f")
    pd.DataFrame({"auc":[real.test_auc],"ap":[real.test_average_precision]}).to_csv(DATA / "scatter_real.dat", sep=" ", index=False, float_format="%.8f")
    print(f"Wrote PGFPlots data files to {DATA}")

if __name__ == "__main__": main()
