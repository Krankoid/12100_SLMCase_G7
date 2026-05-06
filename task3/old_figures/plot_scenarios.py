"""
Generate all Task 3 figures for the executive summary and infographic.

Run from the repo root:
    python task3/figures/plot_scenarios.py

Outputs (in task3/figures/):
    training_co2.pdf   — bar chart, training CO2e by scenario
    inference_co2.pdf  — bar chart, inference CO2e by scenario
    combined.pdf       — side-by-side panel for the infographic
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__))

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ── Data ─────────────────────────────────────────────────────────────────────
TRAIN_LABELS = [
    "Baseline\n(4L/4H/128\nGPU)",
    "Alt A\n(8L/8H/256\nGPU)",
    "Alt B\n(4L/4H/128\nCPU)",
]
TRAIN_CO2_G = [0.568, 2.068, 1.009]

INFER_LABELS = [
    "Baseline\n(100 tokens\nGPU)",
    "Alt A\n(100 tokens\nlarger model\nGPU)",
    "Alt B\n(1000 tokens\nbaseline model\nGPU)",
]
INFER_CO2_MG = [1.14, 2.06, 9.90]

# Colour scheme: baseline = neutral grey, model-size change = blue, hardware change = orange
BLUE = "#2171b5"
ORANGE = "#d94801"
GREY = "#636363"
TRAIN_COLORS = [GREY, BLUE, ORANGE]
INFER_COLORS = [GREY, BLUE, ORANGE]


def _annotate_ratio(ax, bars, values, baseline_val):
    """Add ratio labels above each bar relative to baseline."""
    for bar, val in zip(bars, values):
        ratio = val / baseline_val
        label = "baseline" if ratio == 1.0 else f"{ratio:.1f}×"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * max(values),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


# ── Figure 1: Training CO2e ───────────────────────────────────────────────────
def plot_training_co2():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(TRAIN_LABELS))
    bars = ax.bar(x, TRAIN_CO2_G, color=TRAIN_COLORS, width=0.5, zorder=3)

    _annotate_ratio(ax, bars, TRAIN_CO2_G, TRAIN_CO2_G[0])

    ax.set_xticks(x)
    ax.set_xticklabels(TRAIN_LABELS, fontsize=9)
    ax.set_ylabel("CO₂e (g per training run)")
    ax.set_title("Training emissions by scenario", fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=GREY, label="Baseline"),
        mpatches.Patch(color=BLUE, label="Alt A: model size ↑"),
        mpatches.Patch(color=ORANGE, label="Alt B: CPU instead of GPU"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "training_co2.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 2: Inference CO2e ──────────────────────────────────────────────────
def plot_inference_co2():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = np.arange(len(INFER_LABELS))
    bars = ax.bar(x, INFER_CO2_MG, color=INFER_COLORS, width=0.5, zorder=3)

    _annotate_ratio(ax, bars, INFER_CO2_MG, INFER_CO2_MG[0])

    ax.set_xticks(x)
    ax.set_xticklabels(INFER_LABELS, fontsize=8)
    ax.set_ylabel("CO₂e (mg per response)")
    ax.set_title("Inference emissions by scenario", fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=GREY, label="Baseline"),
        mpatches.Patch(color=BLUE, label="Alt A: model size ↑"),
        mpatches.Patch(color=ORANGE, label="Alt B: token count ↑ (×10)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "inference_co2.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 3: Side-by-side panel (for infographic) ───────────────────────────
def plot_combined():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Training
    x1 = np.arange(len(TRAIN_LABELS))
    bars1 = ax1.bar(x1, TRAIN_CO2_G, color=TRAIN_COLORS, width=0.5, zorder=3)
    _annotate_ratio(ax1, bars1, TRAIN_CO2_G, TRAIN_CO2_G[0])
    ax1.set_xticks(x1)
    ax1.set_xticklabels(TRAIN_LABELS, fontsize=8)
    ax1.set_ylabel("CO₂e (g per training run)")
    ax1.set_title("Training", fontweight="bold")
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.yaxis.grid(True, alpha=0.3)

    # Inference
    x2 = np.arange(len(INFER_LABELS))
    bars2 = ax2.bar(x2, INFER_CO2_MG, color=INFER_COLORS, width=0.5, zorder=3)
    _annotate_ratio(ax2, bars2, INFER_CO2_MG, INFER_CO2_MG[0])
    ax2.set_xticks(x2)
    ax2.set_xticklabels(INFER_LABELS, fontsize=7.5)
    ax2.set_ylabel("CO₂e (mg per response)")
    ax2.set_title("Inference", fontweight="bold")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.yaxis.grid(True, alpha=0.3)

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=GREY, label="Baseline"),
        mpatches.Patch(color=BLUE, label="Alt A: model size ↑"),
        mpatches.Patch(color=ORANGE, label="Alt B: hardware / token count ↑"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("SLM Sustainability — CO₂e emissions by scenario", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    path = os.path.join(OUT_DIR, "combined.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 4: Power / Duration / Energy breakdown (training) ─────────────────
def plot_power_runtime_energy():
    """Three-panel chart showing that high instantaneous power ≠ high total energy.

    Baseline and Alt A use a GPU (high W, short runtime) while Alt B uses a CPU
    (low W, long runtime).  The energy panel is the 'punchline': despite drawing
    30x less power, Alt B consumes more total energy than Baseline because of its
    30x longer runtime.
    """
    scenarios = ["Baseline\n(GPU)", "Alt A\n(GPU, larger)", "Alt B\n(CPU)"]
    # Total system power = GPU + CPU + RAM for each scenario
    # Baseline: 296.7 + 5.0 + 10.0 | Alt A: 353.9 + 4.7 + 10.0 | Alt B: 0.25 + 12.3 + 10.0
    power_w   = [311.7, 368.6, 22.55]   # average W under load (GPU+CPU+RAM combined)
    duration_s = [43.1, 133.3, 1324.9]  # wall-clock seconds
    energy_wh  = [                       # total energy in Wh (kWh × 1000)
        0.003746 * 1000,
        0.013634 * 1000,
        0.006655 * 1000,
    ]

    colors = [GREY, BLUE, ORANGE]
    x = np.arange(len(scenarios))
    bar_w = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    fig.suptitle("Training: power draw vs. runtime vs. total energy", fontweight="bold", fontsize=12)

    # (values, title, ylabel, value_fmt) — value_fmt is used for the absolute label
    datasets = [
        (axes[0], power_w,    "Average power draw (W)",  "Power (W)",    "{:.0f} W"),
        (axes[1], duration_s, "Runtime (s)",              "Duration (s)", "{:.0f} s"),
        (axes[2], energy_wh,  "Total energy (Wh)",        "Energy (Wh)",  "{:.2f} Wh"),
    ]

    for ax, values, title, ylabel, val_fmt in datasets:
        bars = ax.bar(x, values, color=colors, width=bar_w, zorder=3)
        for bar, val in zip(bars, values):
            ratio = val / values[0]
            # ratio label above bar — use 2 decimal places so "0.04×" is readable
            ratio_label = "1×" if ratio == 1.0 else f"{ratio:.2f}×"
            # absolute value label inside (or just above) the bar
            abs_label = val_fmt.format(val)
            bar_top = bar.get_height()
            gap = 0.06 * max(values)   # increased from 0.025 to give more room
            # absolute value: place inside bar if bar is tall enough, else just above
            if bar_top > 0.12 * max(values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_top * 0.5,
                    abs_label,
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_top + gap * 0.3,
                    abs_label,
                    ha="center", va="bottom", fontsize=8, color="black",
                )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar_top + gap,
                ratio_label,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        # pad the y-axis top so the ratio labels don't overlap the subplot title
        ax.set_ylim(top=max(values) * 1.28)
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=8.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "training_power_runtime_energy.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_training_co2()
    plot_inference_co2()
    plot_combined()
    plot_power_runtime_energy()
    print("All figures generated.")
