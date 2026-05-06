"""
Scenario 2 — training model depth sweep.

Sweep N_LAYER in {4, 8, 12} on GPU with everything else at the shared baseline:
  N_HEAD=4, N_EMBD=128, BATCH_SIZE=32, BLOCK_SIZE=256, MAX_ITERS=2000.

N_HEAD=4 divides N_EMBD=128 evenly so all layer counts are valid.
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch  # noqa: E402

SCENARIO = "scenario2_layers"
OUT_DIR = os.path.join("out_lifecycle", SCENARIO)
LAYERS = [4, 8, 12]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_one(n_layer: int):
    print(f"\n{'='*60}\nTRAIN [{SCENARIO}] n_layer={n_layer} device={DEVICE}\n{'='*60}")
    import train_og as T
    importlib.reload(T)

    T.RUN_TAG = f"L{n_layer}"
    T.DEVICE = DEVICE
    T.OUT_DIR = OUT_DIR
    T.N_LAYER = n_layer
    T.N_HEAD = 4
    T.N_EMBD = 128
    T.BATCH_SIZE = 32
    T.BLOCK_SIZE = 256
    T.MAX_ITERS = 2000

    T.main()


def main():
    if DEVICE != "cuda":
        print("WARNING: cuda not available, scenario will run on cpu (slow)")
    os.makedirs(OUT_DIR, exist_ok=True)
    for n_layer in LAYERS:
        run_one(n_layer)
    print(f"\nAll done. Per-stage CSVs in {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
