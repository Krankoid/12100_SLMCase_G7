"""
Three-panel CPU vs GPU training comparison: average power, wall-clock duration,
and total CO2e for the same 2000-iteration training run.

The point of the figure is the counterintuitive result: GPU draws far more
average power, but finishes so much faster that total emissions are *lower*.

Reads:
    out_lifecycle/scenario1_hardware/stages_cpu.csv
    out_lifecycle/scenario1_hardware/stages_cuda.csv

Writes:
    task3/figures/scenario1_hardware_3panel.png

Usage (from project root):
    python task3/figures/plot_hardware_3panel.py
"""

import csv
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCN_DIR = os.path.join(PROJECT_ROOT, "out_lifecycle", "scenario1_hardware")

CPU_COLOR = "#C44E52"
GPU_COLOR = "#4C72B0"


def load_training_loop(path):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["stage"] == "training_loop":
                return {
                    "energy_kwh": float(row["energy_kwh"]),
                    "emissions_kg": float(row["emissions_kg"]),
                    "duration_s": float(row["total_duration_s"]),
                }
    raise RuntimeError(f"training_loop row not found in {path}")


def main():
    cpu = load_training_loop(os.path.join(SCN_DIR, "stages_cpu.csv"))
    gpu = load_training_loop(os.path.join(SCN_DIR, "stages_cuda.csv"))

    avg_power_w = {
        "CPU": cpu["energy_kwh"] * 1000.0 / (cpu["duration_s"] / 3600.0),
        "GPU": gpu["energy_kwh"] * 1000.0 / (gpu["duration_s"] / 3600.0),
    }
    duration_s = {"CPU": cpu["duration_s"], "GPU": gpu["duration_s"]}
    co2_g = {"CPU": cpu["emissions_kg"] * 1000.0, "GPU": gpu["emissions_kg"] * 1000.0}

    labels = ["CPU", "GPU"]
    colors = [CPU_COLOR, GPU_COLOR]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))

    panels = [
        (axes[0], "Average power draw",   "Power (W)",      [avg_power_w[k] for k in labels], "{:.0f} W"),
        (axes[1], "Wall-clock duration",  "Duration (s)",   [duration_s[k]  for k in labels], "{:.0f} s"),
        (axes[2], "Total CO₂e",       "CO₂e (g)",  [co2_g[k]       for k in labels], "{:.2f} g"),
    ]

    for ax, title, ylabel, values, value_fmt in panels:
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ymax = max(values)
        ax.set_ylim(0, ymax * 1.18)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                value_fmt.format(val),
                ha="center", va="bottom", fontsize=10,
            )

    fig.suptitle(
        "Training one SLM checkpoint (2000 iters): runtime, not power draw, decides emissions",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(HERE, "scenario1_hardware_3panel.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    print(f"  CPU: {avg_power_w['CPU']:.1f} W avg, {duration_s['CPU']:.1f} s, {co2_g['CPU']:.3f} g CO2e")
    print(f"  GPU: {avg_power_w['GPU']:.1f} W avg, {duration_s['GPU']:.1f} s, {co2_g['GPU']:.3f} g CO2e")


if __name__ == "__main__":
    main()
