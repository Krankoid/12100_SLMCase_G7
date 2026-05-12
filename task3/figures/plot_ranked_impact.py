"""
Horizontal bar chart ranking the five swept parameters by how much they change
total CO2e from min to max value (max / min ratio).

For training scenarios (1, 2, 3) the metric is total training-loop CO2e per
2000-iter run. For inference scenarios (4, 5) the metric is total per-response
CO2e summed across all stages.

Reads:
    out_lifecycle/scenario1_hardware/stages_{cpu,cuda}.csv
    out_lifecycle/scenario2_layers/stages_L{4,8,12}.csv
    out_lifecycle/scenario3_batchsize/stages_B{32,64,96}.csv
    out_lifecycle/scenario4_temperature/stages_T*.csv
    out_lifecycle/scenario5_tokens/stages_N*.csv

Writes:
    task3/figures/ranked_impact.png

Usage (from project root):
    python task3/figures/plot_ranked_impact.py
"""

import csv
import glob
import os
import re

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_LIFECYCLE = os.path.join(PROJECT_ROOT, "out_lifecycle")


def training_loop_co2(path):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["stage"] == "training_loop":
                return float(row["emissions_kg"])
    raise RuntimeError(f"training_loop row not found in {path}")


def inference_total_co2(path):
    total = 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            total += float(row["emissions_kg"])
    return total


def scenario_ratio(folder, loader, pattern):
    paths = sorted(glob.glob(os.path.join(OUT_LIFECYCLE, folder, pattern)))
    if not paths:
        raise RuntimeError(f"no files matched {folder}/{pattern}")
    values = [loader(p) for p in paths]
    return max(values) / min(values), values


def main():
    rows = [
        ("Hardware (CPU vs GPU)",   *scenario_ratio("scenario1_hardware",    training_loop_co2,   "stages_*.csv")),
        ("Model depth (4-12 layers)",   *scenario_ratio("scenario2_layers",      training_loop_co2,   "stages_L*.csv")),
        ("Batch size (32-96)",      *scenario_ratio("scenario3_batchsize",   training_loop_co2,   "stages_B*.csv")),
        ("Output length (500-1500 tok)", *scenario_ratio("scenario5_tokens",      inference_total_co2, "stages_N*.csv")),
        ("Temperature (0.6-1.4)",   *scenario_ratio("scenario4_temperature", inference_total_co2, "stages_T*.csv")),
    ]
    rows.sort(key=lambda r: r[1], reverse=True)

    labels = [r[0] for r in rows]
    ratios = [r[1] for r in rows]

    color_map = {
        "Hardware (CPU vs GPU)":         "#C44E52",
        "Model depth (4-12 layers)":     "#4C72B0",
        "Batch size (32-96)":            "#937860",
        "Output length (500-1500 tok)":  "#55A467",
        "Temperature (0.6-1.4)":         "#8172B2",
    }
    colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    y_pos = list(range(len(labels)))
    ax.barh(y_pos, ratios, color=colors, edgecolor="white", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(1.0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("CO₂e change ratio (max / min across swept values)")
    ax.set_title("Which parameter moves the needle? Ranked CO₂e impact")

    for y, r in zip(y_pos, ratios):
        ax.text(r + 0.04, y, f"{r:.2f}×", va="center", fontsize=10)

    ax.set_xlim(0, max(ratios) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = os.path.join(HERE, "ranked_impact.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    for label, ratio, values in rows:
        print(f"  {label:32s}  {ratio:.3f}x  values={['%.3e' % v for v in values]}")


if __name__ == "__main__":
    main()
