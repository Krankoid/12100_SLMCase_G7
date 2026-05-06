"""
Scenario 3 — training batch size sweep.

Sweep BATCH_SIZE in {32, 64, 96} on GPU with everything else at the shared baseline:
  N_LAYER=4, N_HEAD=4, N_EMBD=128, BLOCK_SIZE=256, MAX_ITERS=2000.
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch  # noqa: E402

SCENARIO = "scenario3_batchsize"
OUT_DIR = os.path.join("out_lifecycle", SCENARIO)
BATCH_SIZES = [32, 64, 96]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_one(batch_size: int):
    print(f"\n{'='*60}\nTRAIN [{SCENARIO}] batch_size={batch_size} device={DEVICE}\n{'='*60}")
    import train_og as T
    importlib.reload(T)

    T.RUN_TAG = f"B{batch_size}"
    T.DEVICE = DEVICE
    T.OUT_DIR = OUT_DIR
    T.N_LAYER = 4
    T.N_HEAD = 4
    T.N_EMBD = 128
    T.BATCH_SIZE = batch_size
    T.BLOCK_SIZE = 256
    T.MAX_ITERS = 2000

    T.main()


def main():
    if DEVICE != "cuda":
        print("WARNING: cuda not available, scenario will run on cpu (slow)")
    os.makedirs(OUT_DIR, exist_ok=True)
    for batch_size in BATCH_SIZES:
        run_one(batch_size)
    print(f"\nAll done. Per-stage CSVs in {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
