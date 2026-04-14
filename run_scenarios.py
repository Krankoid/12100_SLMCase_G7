"""
Runs all 6 scenarios (3 training + 3 inference) and writes a summary CSV.

Run from inside GWPinLanguageModels-main/ AFTER running:
    python data/prepare.py

Usage:
    conda activate slm-sustainability
    cd GWPinLanguageModels-main
    python run_scenarios.py

Output: out/results_summary.csv
        out/emissions_train_<tag>.csv   (one per training scenario)
        out/emissions_infer_<tag>.csv   (one per inference scenario)
        out/ckpt_<tag>.pt               (one checkpoint per training scenario)
"""

import csv
import importlib
import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC)

import torch  # noqa: E402

HAS_CUDA = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "n/a"

OUT_DIR = "out"
SUMMARY_CSV = os.path.join(OUT_DIR, "results_summary.csv")

# ---------------------------------------------------------------------------
# Scenario definitions (match Task 1 tables)
# ---------------------------------------------------------------------------

TRAIN_SCENARIOS = [
    # (tag,       n_layer, n_head, n_embd, device)
    ("baseline",  4,       4,      128,    "cuda" if HAS_CUDA else "cpu"),
    ("altA",      8,       8,      256,    "cuda" if HAS_CUDA else "cpu"),
    ("altB",      4,       4,      128,    "cpu"),
]

INFER_SCENARIOS = [
    # (tag,       ckpt_tag,    max_new_tokens, num_runs)
    ("baseline",  "baseline",  100,            30),
    ("altA",      "altA",      100,            30),
    ("altB",      "baseline",  1000,           10),
]


def _read_last_codecarbon_row(csv_path):
    """Return (energy_kwh, emissions_kg) from the last row of a CodeCarbon CSV."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    r = rows[-1]
    return float(r.get("energy_consumed", 0)), float(r.get("emissions", 0))


def _ensure_summary():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "phase", "scenario", "device", "layers", "heads", "embd",
                "tokens_or_iters", "num_runs",
                "energy_kwh_total", "energy_kwh_per_run",
                "emissions_kg_total", "emissions_kg_per_run",
            ])


def _append_row(row):
    with open(SUMMARY_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


def run_training(tag, n_layer, n_head, n_embd, device):
    dst_ckpt = os.path.join(OUT_DIR, f"ckpt_{tag}.pt")
    if os.path.exists(dst_ckpt):
        print(f"\nSKIPPING train [{tag}]: checkpoint already exists at {dst_ckpt}")
        print("  Delete it to re-run this scenario.")
        return

    print(f"\n{'='*60}")
    print(f"TRAIN [{tag}]  layers={n_layer}  heads={n_head}  embd={n_embd}  device={device}")
    print(f"{'='*60}")

    import train as T
    importlib.reload(T)

    # Set all module-level globals that train.main() reads
    T.RUN_TAG = tag
    T.N_LAYER = n_layer
    T.N_HEAD = n_head
    T.N_EMBD = n_embd
    T.DEVICE = device
    T.OUT_DIR = OUT_DIR

    T.main()   # train.py creates and stops its own EmissionsTracker internally

    # Rename checkpoint so it is not overwritten by the next scenario
    src_ckpt = os.path.join(OUT_DIR, "ckpt.pt")
    dst_ckpt = os.path.join(OUT_DIR, f"ckpt_{tag}.pt")
    if os.path.exists(src_ckpt):
        os.replace(src_ckpt, dst_ckpt)
        print(f"Checkpoint saved as {dst_ckpt}")

    # Read results from the CSV that CodeCarbon wrote
    cc_csv = os.path.join(OUT_DIR, f"emissions_train_{tag}.csv")
    energy, emissions = _read_last_codecarbon_row(cc_csv)

    _append_row([
        "train", tag, device, n_layer, n_head, n_embd,
        T.MAX_ITERS, 1,
        f"{energy:.6f}", f"{energy:.6f}",
        f"{emissions:.6f}", f"{emissions:.6f}",
    ])
    print(f"[Summary] energy={energy:.6f} kWh  emissions={emissions:.6f} kgCO2e")


def run_inference(tag, ckpt_tag, max_new_tokens, num_runs):
    cc_csv = os.path.join(OUT_DIR, f"emissions_infer_{tag}.csv")
    if os.path.exists(cc_csv):
        print(f"\nSKIPPING infer [{tag}]: results already exist at {cc_csv}")
        print("  Delete it to re-run this scenario.")
        return

    print(f"\n{'='*60}")
    print(f"INFER [{tag}]  ckpt={ckpt_tag}  tokens={max_new_tokens}  runs={num_runs}")
    print(f"{'='*60}")

    ckpt_path = os.path.join(OUT_DIR, f"ckpt_{ckpt_tag}.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint {ckpt_path} not found. Run training first.")
        return

    import prompt as P
    importlib.reload(P)

    P.RUN_TAG = tag
    P.CKPT_PATH = ckpt_path
    P.MAX_NEW_TOKENS = max_new_tokens
    P.NUM_RUNS = num_runs
    P.OUT_DIR = OUT_DIR

    P.main()   # prompt.py creates and stops its own EmissionsTracker internally

    cc_csv = os.path.join(OUT_DIR, f"emissions_infer_{tag}.csv")
    energy, emissions = _read_last_codecarbon_row(cc_csv)

    _append_row([
        "infer", tag, P.DEVICE, "-", "-", "-",
        max_new_tokens, num_runs,
        f"{energy:.6f}", f"{energy / num_runs:.8f}",
        f"{emissions:.6f}", f"{emissions / num_runs:.8f}",
    ])
    print(f"[Summary] energy/run={energy/num_runs:.8f} kWh  "
          f"emissions/run={emissions/num_runs:.8f} kgCO2e")


def main():
    _ensure_summary()
    print(f"CUDA available: {HAS_CUDA}  ({GPU_NAME})")
    print(f"Results will be written to: {os.path.abspath(SUMMARY_CSV)}\n")

    for args in TRAIN_SCENARIOS:
        run_training(*args)

    for args in INFER_SCENARIOS:
        run_inference(*args)

    print(f"\nAll done. Open {SUMMARY_CSV} to see the numbers.")


if __name__ == "__main__":
    main()
