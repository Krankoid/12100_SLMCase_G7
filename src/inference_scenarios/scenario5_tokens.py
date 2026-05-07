"""
Scenario 5 — inference output length sweep.

Sweep MAX_NEW_TOKENS in {500, 750, 1000, 1250, 1500} on GPU with the baseline
trained checkpoint. TEMPERATURE is fixed at 1.0 (the baseline).
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch  # noqa: E402

SCENARIO = "scenario5_tokens"
OUT_DIR = os.path.join("out_lifecycle", SCENARIO)
CKPT_PATH = os.path.join("out_lifecycle", "checkpoints", "ckpt_baseline.pt")

TOKEN_COUNTS = [500, 750, 1000, 1250, 1500]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_one(max_new_tokens: int):
    tag = f"N{max_new_tokens}"
    print(f"\n{'='*60}\nINFER [{SCENARIO}] max_new_tokens={max_new_tokens} device={DEVICE}\n{'='*60}")
    import prompt_og as P
    importlib.reload(P)

    P.RUN_TAG = tag
    P.DEVICE = DEVICE
    P.OUT_DIR = OUT_DIR
    P.CKPT_PATH = CKPT_PATH
    P.MAX_NEW_TOKENS = max_new_tokens
    P.TEMPERATURE = 1.0
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
    # first token-count point isn't biased by cuDNN autotune for long sequences.
    import prompt_og as P
    P.gpu_warmup(max_new_tokens=max(TOKEN_COUNTS), ckpt_path=CKPT_PATH, device=DEVICE)

    for n in TOKEN_COUNTS:
        run_one(n)
    print(f"\nAll done. Per-stage CSVs in {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
