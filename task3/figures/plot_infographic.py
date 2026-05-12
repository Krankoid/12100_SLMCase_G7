"""
Generate infographic-ready versions of the six figures used in the executive
summary one-pager.

Layout the user is composing in Figma:
    Top row:    scenario1_hardware  -> duration | power | CO2
    Middle row: scenario2_layers CO2 | scenario5_tokens CO2
    Bottom row: ranked_impact

Differences vs. the originals in task3/figures/:
- No chart titles (captions added manually in Figma)
- Narrower bars on the three hardware plots so they sit comfortably in a row
- Per-plot legends removed; standalone legend PNGs are exported separately
  so a single legend can be placed once in the infographic
- CPU / CUDA tick labels replaced with the actual hardware names
  (Intel i7-10750H / NVIDIA RTX 3090) per the Task 2 hand-in
- Ranked impact bars recoloured on a yellow->red gradient (yellow = least
  impact, red = most impact) so they no longer clash with the life-cycle
  stage palette

Reads:
    out_lifecycle/scenario1_hardware/stages_{cpu,cuda}.csv
    out_lifecycle/scenario2_layers/stages_L{4,8,12}.csv
    out_lifecycle/scenario3_batchsize/stages_B{32,64,96}.csv
    out_lifecycle/scenario4_temperature/stages_T*.csv
    out_lifecycle/scenario5_tokens/stages_N*.csv

Writes everything into task3/figures/infographic_plots/.

Usage (from project root):
    python task3/figures/plot_infographic.py
"""

import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_LIFECYCLE = os.path.join(PROJECT_ROOT, "out_lifecycle")
OUT_DIR = os.path.join(HERE, "infographic_plots")
os.makedirs(OUT_DIR, exist_ok=True)

TRAINING_STAGES = ["setup", "forward", "backward_update", "evaluation"]
INFERENCE_STAGES = ["setup", "encoding", "generation", "decoding"]

STAGE_COLORS = {
    # Training stages
    "setup":            "#4C72B0",  # blue (shared between training & inference)
    "forward":          "#55A467",  # green
    "backward_update":  "#C44E52",  # red
    "evaluation":       "#8172B2",  # purple
    # Inference stages — distinct hues so they never collide with training colors
    "encoding":         "#DD8452",  # orange
    "generation":       "#64B5CD",  # cyan
    "decoding":         "#CCB974",  # mustard
}

DEVICE_TICK_LABELS = {
    "cpu":  "Intel\ni7-10750H",
    "cuda": "NVIDIA\nRTX 3090",
}
DEVICE_BAR_COLORS = {"cpu": "#C44E52", "cuda": "#4C72B0"}

HARDWARE_FIGSIZE = (3.4, 4.2)
HARDWARE_BAR_WIDTH = 0.9
LAYER_TOKEN_FIGSIZE = (5.2, 4.2)
RANKED_FIGSIZE = (8.4, 3.6)


def load_stages(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["stage"]] = {
                "energy_kwh": float(r["energy_kwh"]),
                "emissions_kg": float(r["emissions_kg"]),
                "duration_s": float(r["total_duration_s"]),
            }
    return rows


def style_axes(ax):
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    out_path = os.path.join(OUT_DIR, name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# --------------------------------------------------------------------------
# Top row: three hardware plots (duration, power, CO2)
# --------------------------------------------------------------------------

def plot_hardware_stacked(metric, ylabel, fname):
    """Stacked bar across life-cycle stages, one bar per device."""
    devices = ["cpu", "cuda"]
    runs = {d: load_stages(os.path.join(OUT_LIFECYCLE, "scenario1_hardware",
                                        f"stages_{d}.csv")) for d in devices}

    fig, ax = plt.subplots(figsize=HARDWARE_FIGSIZE)
    x_pos = list(range(len(devices)))
    bottoms = [0.0] * len(devices)
    for stage in TRAINING_STAGES:
        heights = [runs[d].get(stage, {}).get(metric, 0.0) for d in devices]
        ax.bar(x_pos, heights, width=HARDWARE_BAR_WIDTH, bottom=bottoms,
               color=STAGE_COLORS[stage], edgecolor="white", linewidth=0.5)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(x_pos)
    ax.set_xticklabels([DEVICE_TICK_LABELS[d] for d in devices], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.margins(x=0.35)
    style_axes(ax)
    fig.tight_layout()
    save(fig, fname)


def plot_hardware_power():
    """Single bar per device for duration-weighted average power."""
    devices = ["cpu", "cuda"]
    powers = []
    for d in devices:
        s = load_stages(os.path.join(OUT_LIFECYCLE, "scenario1_hardware",
                                     f"stages_{d}.csv"))
        total_kwh = sum(v["energy_kwh"] for v in s.values())
        total_s = sum(v["duration_s"] for v in s.values())
        powers.append(total_kwh * 1000.0 / (total_s / 3600.0) if total_s else 0.0)

    fig, ax = plt.subplots(figsize=HARDWARE_FIGSIZE)
    x_pos = list(range(len(devices)))
    bars = ax.bar(x_pos, powers, width=HARDWARE_BAR_WIDTH,
                  color=[DEVICE_BAR_COLORS[d] for d in devices],
                  edgecolor="white", linewidth=0.5)
    for b, h in zip(bars, powers):
        ax.text(b.get_x() + b.get_width() / 2, h,
                f"{h:.0f} W", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([DEVICE_TICK_LABELS[d] for d in devices], fontsize=9)
    ax.set_ylabel("Average power draw (W)")
    ax.margins(x=0.35, y=0.18)
    style_axes(ax)
    fig.tight_layout()
    save(fig, "hardware_power.png")


# --------------------------------------------------------------------------
# Middle row: layers CO2 and tokens CO2
# --------------------------------------------------------------------------

def plot_sweep_stacked(folder, phase, x_label, sort_key, tag_to_label, fname):
    stages = TRAINING_STAGES if phase == "training" else INFERENCE_STAGES
    runs = {}
    for path in glob.glob(os.path.join(OUT_LIFECYCLE, folder, "stages_*.csv")):
        tag = re.match(r"^stages_(.+)\.csv$", os.path.basename(path)).group(1)
        runs[tag] = load_stages(path)

    sorted_tags = sorted(runs.keys(), key=sort_key)
    x_pos = list(range(len(sorted_tags)))

    fig, ax = plt.subplots(figsize=LAYER_TOKEN_FIGSIZE)
    bottoms = [0.0] * len(sorted_tags)
    for stage in stages:
        heights = [runs[t].get(stage, {}).get("emissions_kg", 0.0) for t in sorted_tags]
        ax.bar(x_pos, heights, bottom=bottoms,
               color=STAGE_COLORS[stage], edgecolor="white", linewidth=0.5)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(x_pos)
    ax.set_xticklabels([tag_to_label(t) for t in sorted_tags])
    ax.set_xlabel(x_label)
    ax.set_ylabel("CO₂ emissions (kg)")
    style_axes(ax)
    fig.tight_layout()
    save(fig, fname)


# --------------------------------------------------------------------------
# Bottom row: ranked impact, yellow -> red gradient
# --------------------------------------------------------------------------

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
    values = [loader(p) for p in paths]
    return max(values) / min(values)


def plot_ranked_impact():
    rows = [
        ("Hardware (CPU vs GPU)",         scenario_ratio("scenario1_hardware",    training_loop_co2,   "stages_*.csv")),
        ("Model depth (4-12 layers)",     scenario_ratio("scenario2_layers",      training_loop_co2,   "stages_L*.csv")),
        ("Batch size (32-96)",            scenario_ratio("scenario3_batchsize",   training_loop_co2,   "stages_B*.csv")),
        ("Output length (500-1500 tok)",  scenario_ratio("scenario5_tokens",      inference_total_co2, "stages_N*.csv")),
        ("Temperature (0.6-1.4)",         scenario_ratio("scenario4_temperature", inference_total_co2, "stages_T*.csv")),
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    ratios = [r[1] for r in rows]

    # Yellow -> red gradient: highest ratio gets the deepest red, lowest gets
    # warm yellow. Bars are drawn top-to-bottom in descending order, so the
    # gradient runs from red (top) to yellow (bottom).
    cmap = LinearSegmentedColormap.from_list(
        "yl_rd_infographic",
        ["#F4C430", "#E97A2A", "#C8321F"],   # warm yellow -> orange -> deep red
    )
    n = len(ratios)
    colors = [cmap(1.0 - i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=RANKED_FIGSIZE)
    y_pos = list(range(n))
    ax.barh(y_pos, ratios, color=colors, edgecolor="white", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(1.0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("CO₂e change ratio (max / min across swept values)")
    for y, r in zip(y_pos, ratios):
        ax.text(r + 0.04, y, f"{r:.2f}×", va="center", fontsize=10)
    ax.set_xlim(0, max(ratios) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "ranked_impact.png")


# --------------------------------------------------------------------------
# Standalone legend PNGs (one per stage palette)
# --------------------------------------------------------------------------

def _legend_figure(stages, fname, ncol):
    handles = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in stages]
    fig = plt.figure(figsize=(ncol * 1.6, 0.6))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False, fontsize=10)
    save(fig, fname)


def plot_legends():
    # One unified legend that covers every life-cycle stage we plot. Now that
    # training and inference stages use disjoint colors, a single key works
    # for the whole infographic.
    unified = TRAINING_STAGES + [s for s in INFERENCE_STAGES if s not in TRAINING_STAGES]
    _legend_figure(unified, "legend_all_stages.png", ncol=len(unified))
    # Keep the split legends around in case a tighter layout is preferred.
    _legend_figure(TRAINING_STAGES, "legend_training_stages.png", ncol=len(TRAINING_STAGES))
    _legend_figure(INFERENCE_STAGES, "legend_inference_stages.png", ncol=len(INFERENCE_STAGES))


# --------------------------------------------------------------------------
# Suggested captions (for manual placement in Figma)
# --------------------------------------------------------------------------

CAPTIONS = [
    ("hardware_duration.png", "Wall-clock training time per life-cycle stage (CPU vs GPU)."),
    ("hardware_power.png",    "Average power draw during training (CPU vs GPU)."),
    ("hardware_co2.png",      "Training CO₂ emissions per life-cycle stage (CPU vs GPU)."),
    ("layers_co2.png",        "Training CO₂ emissions scale with model depth."),
    ("tokens_co2.png",        "Inference CO₂ emissions scale with output length; generation dominates."),
    ("ranked_impact.png",     "Which knob moves the needle? Max/min CO₂e ratio across each sweep."),
]


def write_captions():
    out_path = os.path.join(OUT_DIR, "captions.txt")
    with open(out_path, "w") as f:
        for name, cap in CAPTIONS:
            f.write(f"{name}\t{cap}\n")
    print(f"[plot] wrote {out_path}")


# --------------------------------------------------------------------------

def main():
    # Top row
    plot_hardware_stacked("duration_s",   "Wall-clock duration (s)",  "hardware_duration.png")
    plot_hardware_power()
    plot_hardware_stacked("emissions_kg", "CO₂ emissions (kg)",  "hardware_co2.png")

    # Middle row
    plot_sweep_stacked(
        folder="scenario2_layers", phase="training",
        x_label="Number of layers",
        sort_key=lambda tag: int(tag[1:]),
        tag_to_label=lambda tag: tag[1:],
        fname="layers_co2.png",
    )
    plot_sweep_stacked(
        folder="scenario5_tokens", phase="inference",
        x_label="Max new tokens",
        sort_key=lambda tag: int(tag[1:]),
        tag_to_label=lambda tag: tag[1:],
        fname="tokens_co2.png",
    )

    # Bottom row
    plot_ranked_impact()

    # Shared legends + caption sheet
    plot_legends()
    write_captions()


if __name__ == "__main__":
    main()
