"""
Scenario 4 — inference temperature sweep.

Sweep TEMPERATURE in {0.6, 0.8, 1.0, 1.2, 1.4} on GPU with the baseline
trained checkpoint. MAX_NEW_TOKENS is fixed at 500 (the new inference baseline)
so a single run resolves above CodeCarbon's sampling floor.
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch  # noqa: E402

SCENARIO = "scenario4_temperature"
OUT_DIR = os.path.join("out_lifecycle", SCENARIO)
CKPT_PATH = os.path.join("out_lifecycle", "checkpoints", "ckpt_baseline.pt")

TEMPERATURES = [0.6, 0.8, 1.0, 1.2, 1.4]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_one(temperature: float):
    tag = f"T{temperature:.1f}".replace(".", "_")
    print(f"\n{'='*60}\nINFER [{SCENARIO}] temperature={temperature} device={DEVICE}\n{'='*60}")
    import prompt_og as P
    importlib.reload(P)

    P.RUN_TAG = tag
    P.DEVICE = DEVICE
    P.OUT_DIR = OUT_DIR
    P.CKPT_PATH = CKPT_PATH
    P.MAX_NEW_TOKENS = 500
    P.TEMPERATURE = temperature
    P.TOP_K = 50

    P.main()


def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {CKPT_PATH}. "
            "Run src/training_scenarios/scenario1_hardware.py first (or scenario2_layers.py "
            "L=4 run) to produce it."
        )
    if DEVICE != "cuda":
        print("WARNING: cuda not available, scenario will run on cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    # one-time GPU warmup at the max token count used in this sweep, so the
    # first temperature point isn't biased by cuDNN autotune for long sequences.
    import prompt_og as P
    P.gpu_warmup(max_new_tokens=500, ckpt_path=CKPT_PATH, device=DEVICE)

    for temperature in TEMPERATURES:
        run_one(temperature)
    print(f"\nAll done. Per-stage CSVs in {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
