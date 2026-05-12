"""
Average power draw (W) for scenario1_hardware (CPU vs CUDA), one bar per
device. Computed as a duration-weighted average across all life-cycle
stages: total energy / total wall-clock time.

Reads out_lifecycle/scenario1_hardware/stages_{cpu,cuda}.csv and writes
task3/figures/scenario1_hardware_power.png.
"""

import csv
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCEN_DIR = os.path.join(PROJECT_ROOT, "out_lifecycle", "scenario1_hardware")
FIG_DIR = HERE

DEVICES = [("cpu", "CPU", "#C44E52"), ("cuda", "CUDA", "#4C72B0")]


def load_stages(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["stage"]] = {
                "energy_kwh": float(r["energy_kwh"]),
                "duration_s": float(r["total_duration_s"]),
            }
    return rows


def overall_avg_power_w(stages):
    total_kwh = sum(s["energy_kwh"] for s in stages.values())
    total_s = sum(s["duration_s"] for s in stages.values())
    if total_s <= 0:
        return 0.0
    return total_kwh * 1000.0 / (total_s / 3600.0)


def main():
    powers = []
    labels = []
    colors = []
    for tag, label, color in DEVICES:
        stages = load_stages(os.path.join(SCEN_DIR, f"stages_{tag}.csv"))
        powers.append(overall_avg_power_w(stages))
        labels.append(label)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, powers, color=colors, edgecolor="white", linewidth=0.5)
    for b, h in zip(bars, powers):
        ax.text(b.get_x() + b.get_width() / 2, h,
                f"{h:.0f} W", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Device")
    ax.set_ylabel("Average power draw (W)")
    ax.set_title("scenario1_hardware — Average power draw (W)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.margins(y=0.15)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "scenario1_hardware_power.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
