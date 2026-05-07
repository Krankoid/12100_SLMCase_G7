"""
Build stacked bar charts (energy + CO2) for each scenario, where each bar is a
single sweep value and each stack segment is a life-cycle stage.

Reads per-stage CSVs from out_lifecycle/<scenario>/stages_*.csv (written by
the instrumented train_og.py and prompt_og.py).

Outputs PNGs to task3/figures/.

Usage (from project root):
    python task3/figures/plot_lifecycle_stacks.py
"""

import csv
import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_LIFECYCLE = os.path.join(PROJECT_ROOT, "out_lifecycle")
FIG_DIR = HERE

TRAINING_STAGES = ["setup", "forward", "backward_update", "evaluation"]
INFERENCE_STAGES = ["setup", "encoding", "generation", "decoding"]

STAGE_COLORS = {
    "setup":            "#4C72B0",
    "data_load":        "#DD8452",
    "forward":          "#55A467",
    "backward_update":  "#C44E52",
    "evaluation":       "#8172B2",
    "checkpoint_save":  "#937860",
    "encoding":         "#DD8452",
    "generation":       "#55A467",
    "decoding":         "#C44E52",
}

SCENARIOS = [
    # (folder, phase, x_label, sort_key)
    ("scenario1_hardware",    "training",  "Device",                   lambda tag: 0 if tag == "cpu" else 1),
    ("scenario2_layers",      "training",  "Number of layers",         lambda tag: int(tag[1:])),                       # L4 -> 4
    ("scenario3_batchsize",   "training",  "Batch size",               lambda tag: int(tag[1:])),                       # B32 -> 32
    ("scenario4_temperature", "inference", "Temperature",              lambda tag: float(tag[1:].replace("_", "."))),    # T0_6 -> 0.6
    ("scenario5_tokens",      "inference", "Max new tokens",           lambda tag: int(tag[1:])),                       # N500 -> 500
]


def load_stages_csv(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["stage"]] = {
                "energy_kwh": float(row["energy_kwh"]),
                "emissions_kg": float(row["emissions_kg"]),
            }
    return out


def collect_scenario(folder):
    """Return OrderedDict of {tag: {stage: {energy_kwh, emissions_kg}}} for one scenario folder."""
    scenario_dir = os.path.join(OUT_LIFECYCLE, folder)
    if not os.path.isdir(scenario_dir):
        return None
    runs = {}
    pat = re.compile(r"^stages_(.+)\.csv$")
    for name in os.listdir(scenario_dir):
        m = pat.match(name)
        if not m:
            continue
        tag = m.group(1)
        runs[tag] = load_stages_csv(os.path.join(scenario_dir, name))
    return runs


def pretty_xlabel(tag, scenario_folder):
    if scenario_folder == "scenario4_temperature":
        return tag[1:].replace("_", ".")
    if scenario_folder.startswith("scenario1"):
        return tag.upper()
    if tag and tag[0].isalpha():
        return tag[1:]
    return tag


def plot_scenario(folder, phase, x_label, sort_key, runs):
    stages = TRAINING_STAGES if phase == "training" else INFERENCE_STAGES

    sorted_tags = sorted(runs.keys(), key=sort_key)
    x_pos = list(range(len(sorted_tags)))
    x_labels = [pretty_xlabel(t, folder) for t in sorted_tags]

    for metric, ylabel, suffix in [
        ("energy_kwh",   "Energy (kWh)",                "kwh"),
        ("emissions_kg", "CO₂ emissions (kg)",     "co2"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bottoms = [0.0] * len(sorted_tags)
        for stage in stages:
            heights = [runs[t].get(stage, {}).get(metric, 0.0) for t in sorted_tags]
            ax.bar(x_pos, heights, bottom=bottoms,
                   color=STAGE_COLORS.get(stage, "#666"),
                   label=stage, edgecolor="white", linewidth=0.5)
            bottoms = [b + h for b, h in zip(bottoms, heights)]

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{folder} — {ylabel} per life-cycle stage")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=False)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()

        out_path = os.path.join(FIG_DIR, f"{folder}_{suffix}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


def main():
    for folder, phase, x_label, sort_key in SCENARIOS:
        runs = collect_scenario(folder)
        if not runs:
            print(f"[skip] {folder}: no stages_*.csv found in {os.path.join(OUT_LIFECYCLE, folder)}")
            continue
        plot_scenario(folder, phase, x_label, sort_key, runs)


if __name__ == "__main__":
    main()
