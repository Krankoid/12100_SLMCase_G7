"""
EcoLogits benchmark — Llama 2 7B vs. our SLM (scenario 5: output-length sweep).

For each MAX_NEW_TOKENS in {500, 750, 1000, 1250, 1500}:
  - read our measured per-stage energy / GWP from out_lifecycle/scenario5_tokens/
  - call EcoLogits compute_llm_impacts for Llama 2 7B with the same output length
  - write a side-by-side comparison CSV + print a summary table

EcoLogits assumptions baked in (v0.10.x):
  - 16-bit quantization, A100-80GB, 8 GPUs/server, BATCH_SIZE=64
  - Electricity mix: Denmark (matches our CodeCarbon report)
  - Datacenter PUE = 1.2 (typical hyperscale; CodeCarbon used PUE=1.0)

Install:
    pip install ecologits

Run from project root (12100_SLMCase_G7/):
    python task3/ecologits_benchmark.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from ecologits.electricity_mix_repository import electricity_mixes
from ecologits.impacts.llm import compute_llm_impacts

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
SCEN5_DIR = PROJ / "out_lifecycle" / "scenario5_tokens"
OUT_CSV = HERE / "ecologits_benchmark.csv"
OUT_PNG = HERE / "figures" / "ecologits_benchmark.png"

SLM_COLOR = "#C44E52"   # red
EL_COLOR = "#4C72B0"    # blue
EL_OP_COLOR = "#8DA0CB"  # lighter blue

TOKEN_COUNTS = [500, 750, 1000, 1250, 1500]

# Llama 2 7B is a dense model: active == total params (in billions).
MODEL_NAME = "Llama 2 7B"
ACTIVE_PARAMS_B = 7.0
TOTAL_PARAMS_B = 7.0

ELEC_ZONE = "DNK"          # match the SLM run's grid mix
DATACENTER_PUE = 1.2       # typical hyperscale
DATACENTER_WUE = 1.8       # L/kWh, typical hyperscale


def read_slm_stages(n_tokens: int):
    """Return dict of stage -> (energy_kwh, emissions_kg) and totals."""
    path = SCEN5_DIR / f"stages_N{n_tokens}.csv"
    stages = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            stages[row["stage"]] = (
                float(row["energy_kwh"]),
                float(row["emissions_kg"]),
            )
    total_kwh = sum(v[0] for v in stages.values())
    total_kg = sum(v[1] for v in stages.values())
    # Inference-only excludes the one-off `setup` stage (model load); EcoLogits
    # amortises model load across many requests, so this is the fairer total.
    infer_kwh = sum(v[0] for k, v in stages.items() if k != "setup")
    infer_kg = sum(v[1] for k, v in stages.items() if k != "setup")
    return {
        "stages": stages,
        "total_kwh": total_kwh,
        "total_kg": total_kg,
        "infer_kwh": infer_kwh,
        "infer_kg": infer_kg,
    }


def ecologits_for(n_tokens: int, mix):
    imp = compute_llm_impacts(
        model_active_parameter_count=ACTIVE_PARAMS_B,
        model_total_parameter_count=TOTAL_PARAMS_B,
        output_token_count=n_tokens,
        if_electricity_mix_adpe=mix.adpe,
        if_electricity_mix_pe=mix.pe,
        if_electricity_mix_gwp=mix.gwp,
        if_electricity_mix_wue=mix.wue,
        datacenter_pue=DATACENTER_PUE,
        datacenter_wue=DATACENTER_WUE,
    )
    return {
        "energy_kwh": float(imp.energy.value),
        "gwp_kg": float(imp.gwp.value),                  # usage + embodied
        "gwp_usage_kg": float(imp.usage.gwp.value),      # operational only
        "gwp_embodied_kg": float(imp.embodied.gwp.value),
    }


def main():
    if not SCEN5_DIR.exists():
        sys.exit(f"Missing data dir: {SCEN5_DIR}")

    mix = electricity_mixes.find_electricity_mix(zone=ELEC_ZONE)
    print(f"Model        : {MODEL_NAME} (dense, {TOTAL_PARAMS_B}B params)")
    print(f"Electricity  : {mix.zone}  GWP={mix.gwp:.4f} kgCO2eq/kWh")
    print(f"Datacenter   : PUE={DATACENTER_PUE}  WUE={DATACENTER_WUE} L/kWh")
    print()

    rows = []
    for n in TOKEN_COUNTS:
        slm = read_slm_stages(n)
        el = ecologits_for(n, mix)
        rows.append({
            "max_new_tokens": n,
            "slm_total_kwh": slm["total_kwh"],
            "slm_total_kg": slm["total_kg"],
            "slm_infer_kwh": slm["infer_kwh"],
            "slm_infer_kg": slm["infer_kg"],
            "el_energy_kwh": el["energy_kwh"],
            "el_gwp_usage_kg": el["gwp_usage_kg"],
            "el_gwp_embodied_kg": el["gwp_embodied_kg"],
            "el_gwp_total_kg": el["gwp_kg"],
            "ratio_energy_el_over_slm": el["energy_kwh"] / slm["infer_kwh"],
            "ratio_gwp_el_over_slm": el["gwp_kg"] / slm["infer_kg"],
        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Pretty summary
    print(f"{'N':>5} | {'SLM kWh (infer)':>16} {'SLM kg (infer)':>16} | "
          f"{'EL kWh':>12} {'EL kg (op)':>12} {'EL kg (tot)':>12} | "
          f"{'kWh ratio':>10} {'kg ratio':>10}")
    print("-" * 120)
    for r in rows:
        print(f"{r['max_new_tokens']:>5} | "
              f"{r['slm_infer_kwh']:>16.3e} {r['slm_infer_kg']:>16.3e} | "
              f"{r['el_energy_kwh']:>12.3e} {r['el_gwp_usage_kg']:>12.3e} "
              f"{r['el_gwp_total_kg']:>12.3e} | "
              f"{r['ratio_energy_el_over_slm']:>10.1f} "
              f"{r['ratio_gwp_el_over_slm']:>10.1f}")

    # Slope (linear fit slope of GWP vs. tokens): a quick sanity check.
    def slope(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        return num / den
    xs = [r["max_new_tokens"] for r in rows]
    slm_slope = slope(xs, [r["slm_infer_kg"] for r in rows])
    el_slope_op = slope(xs, [r["el_gwp_usage_kg"] for r in rows])
    el_slope_tot = slope(xs, [r["el_gwp_total_kg"] for r in rows])
    print()
    print(f"SLM   GWP slope:                 {slm_slope:.3e} kg CO2e / token")
    print(f"EL    GWP slope (operational):   {el_slope_op:.3e} kg CO2e / token  "
          f"(ratio {el_slope_op/slm_slope:.2f}x)")
    print(f"EL    GWP slope (op + embodied): {el_slope_tot:.3e} kg CO2e / token  "
          f"(ratio {el_slope_tot/slm_slope:.2f}x)")
    print()
    print(f"Wrote {OUT_CSV}")

    make_plot(rows, slm_slope, el_slope_op)
    print(f"Wrote {OUT_PNG}")


def make_plot(rows, slm_slope, el_slope_op):
    import matplotlib.pyplot as plt

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    xs = [r["max_new_tokens"] for r in rows]
    slm = [r["slm_infer_kg"] * 1e6 for r in rows]      # µg
    el_op = [r["el_gwp_usage_kg"] * 1e6 for r in rows]
    el_tot = [r["el_gwp_total_kg"] * 1e6 for r in rows]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 2.9),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )

    # Left panel — GWP vs output tokens.
    # Operational is the apples-to-apples line vs. CodeCarbon (which only
    # measures wall electricity). Operational + embodied is shown as the
    # secondary, dashed line: it's what a full life-cycle estimate would
    # add for Llama 2 7B, but our SLM number does not include any
    # embodied counterpart.
    ax1.plot(xs, slm, marker="o", color=SLM_COLOR, lw=1.8,
             label="SLM measured, operational (our run)")
    ax1.plot(xs, el_op, marker="s", color=EL_COLOR, lw=1.8,
             label="Llama 2 7B / EcoLogits, operational")
    ax1.plot(xs, el_tot, marker="^", color=EL_OP_COLOR, lw=1.2, ls="--",
             label="Llama 2 7B / EcoLogits, operational + embodied")
    ax1.set_xlabel("MAX_NEW_TOKENS")
    ax1.set_ylabel(r"GWP per request ($\mu$g CO$_2$e)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.set_xticks(xs)

    # Right panel — per-token marginal cost (the load-bearing comparison),
    # rescaled to µg per 1000 tokens so the numbers read cleanly.
    cats = ["SLM\nmeasured", "Llama 2 7B\nEcoLogits\n(operational)"]
    vals = [slm_slope * 1e9, el_slope_op * 1e9]   # kg/tok -> µg/1000 tok
    colors = [SLM_COLOR, EL_COLOR]
    bars = ax2.bar(cats, vals, color=colors, width=0.55)
    ax2.set_ylabel(r"$\mu$g CO$_2$e per 1 000 tokens")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, max(vals) * 1.35)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.text(0.5, 0.93, f"slope ratio {el_slope_op/slm_slope:.2f}×",
             transform=ax2.transAxes, ha="center", fontsize=9,
             style="italic", color="dimgray")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
