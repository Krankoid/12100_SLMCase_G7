"""
Inference / prompting script (Tiny Shakespeare, char-level) — instrumented with
per-stage CodeCarbon tracking for the life-cycle assessment in
task3/lifecycle_fixes.md.

The inference phase is split into four life-cycle stages:
  Part 1 — setup     : load checkpoint, build model, copy weights, eval()
  Part 2 — encoding  : prompt string -> token tensor on device
  Part 3 — generation: model.generate autoregressive forward loop
  Part 4 — decoding  : token tensor -> output string

Each stage is wrapped in tracker.start_task("<tag>") / tracker.stop_task().
A single run per script invocation; no inner repetition loop. The driver
scripts in src/inference_scenarios/ set MAX_NEW_TOKENS large enough (>=500)
that one run resolves above CodeCarbon's sampling floor.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import csv
import pickle
import torch
from codecarbon import EmissionsTracker

from model import GPT, GPTConfig

# ----------------------------
# Edit these (overridden by per-scenario driver scripts)
# ----------------------------
OUT_DIR = "out_lifecycle/scenario_default"
CKPT_PATH = "out_lifecycle/checkpoints/ckpt_baseline.pt"

PROMPT = "To be, or not to be"
MAX_NEW_TOKENS = 500
TEMPERATURE = 1.0
TOP_K = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_TAG = "default"
# ----------------------------


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def _record(stage_totals, stage, emissions_data):
    if emissions_data is None:
        return
    stage_totals[stage]["energy_kwh"] += emissions_data.energy_consumed
    stage_totals[stage]["emissions_kg"] += emissions_data.emissions
    stage_totals[stage]["duration_s"] += emissions_data.duration
    stage_totals[stage]["n_calls"] += 1


def _write_stages_csv(out_dir: str, run_tag: str, stage_totals: dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"stages_{run_tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "energy_kwh", "emissions_kg", "n_calls", "total_duration_s", "mean_call_duration_s"])
        for stage, d in stage_totals.items():
            mean = d["duration_s"] / d["n_calls"] if d["n_calls"] else 0.0
            w.writerow([stage, f"{d['energy_kwh']:.10f}", f"{d['emissions_kg']:.10f}",
                        d["n_calls"], f"{d['duration_s']:.6f}", f"{mean:.6f}"])
    print(f"[stages] wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tracker = EmissionsTracker(
        project_name=f"infer_{RUN_TAG}_tokens{MAX_NEW_TOKENS}_temp{TEMPERATURE}",
        output_dir=OUT_DIR,
        output_file=f"codecarbon_infer_{RUN_TAG}.csv",
        measure_power_secs=1,
        log_level="warning",
    )
    tracker.start()

    stage_totals = {
        s: {"energy_kwh": 0.0, "emissions_kg": 0.0, "duration_s": 0.0, "n_calls": 0}
        for s in ["setup", "encoding", "generation", "decoding"]
    }

    # ------------------------------------------------------------------
    # Part 1 — setup: load checkpoint, build model, copy weights to device
    # ------------------------------------------------------------------
    tracker.start_task("setup")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]

    meta = load_meta(data_dir)
    stoi = meta["stoi"]
    itos = meta["itos"]

    config = GPTConfig(**model_cfg)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _record(stage_totals, "setup", tracker.stop_task())

    def encode(s: str):
        return [stoi.get(ch, stoi[" "]) for ch in s]

    def decode(tokens):
        return "".join([itos[t] for t in tokens])

    # ------------------------------------------------------------------
    # Part 2 — encoding: prompt string -> token tensor on device
    # ------------------------------------------------------------------
    tracker.start_task("encoding")
    idx = torch.tensor([encode(PROMPT)], dtype=torch.long, device=DEVICE)
    _record(stage_totals, "encoding", tracker.stop_task())

    # ------------------------------------------------------------------
    # Part 3 — generation: autoregressive forward loop, MAX_NEW_TOKENS new tokens
    # ------------------------------------------------------------------
    tracker.start_task("generation")
    out = model.generate(
        idx,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )
    _record(stage_totals, "generation", tracker.stop_task())

    # ------------------------------------------------------------------
    # Part 4 — decoding: token tensor -> output string
    # ------------------------------------------------------------------
    tracker.start_task("decoding")
    text = decode(out[0].tolist())
    _record(stage_totals, "decoding", tracker.stop_task())

    tracker.stop()
    _write_stages_csv(OUT_DIR, RUN_TAG, stage_totals)

    print(text)
    total_kwh = sum(d["energy_kwh"] for d in stage_totals.values())
    total_kg = sum(d["emissions_kg"] for d in stage_totals.values())
    print(f"\n[total] tag={RUN_TAG} device={DEVICE} tokens={MAX_NEW_TOKENS} "
          f"temp={TEMPERATURE} energy={total_kwh:.8f} kWh emissions={total_kg:.8f} kgCO2e")


if __name__ == "__main__":
    main()
