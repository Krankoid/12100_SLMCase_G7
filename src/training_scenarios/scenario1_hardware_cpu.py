"""
Scenario 1 — CPU half only.

Trains the baseline configuration (N_LAYER=4, N_HEAD=4, N_EMBD=128,
BATCH_SIZE=32, BLOCK_SIZE=256, MAX_ITERS=2000) on cpu.

Outputs go to out_lifecycle/scenario1_hardware/ — same folder as the gpu
half, so plotting code that reads that folder will see both runs once both
scripts have completed.

Run:
    python -m src.training_scenarios.scenario1_hardware_cpu
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

SCENARIO = "scenario1_hardware"
OUT_DIR = os.path.join("out_lifecycle", SCENARIO)
DEVICE = "cpu"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*60}\nTRAIN [{SCENARIO}] device={DEVICE}\n{'='*60}")
    import train_og as T
    importlib.reload(T)

    T.RUN_TAG = DEVICE
    T.DEVICE = DEVICE
    T.OUT_DIR = OUT_DIR
    T.N_LAYER = 4
    T.N_HEAD = 4
    T.N_EMBD = 128
    T.BATCH_SIZE = 32
    T.BLOCK_SIZE = 256
    T.MAX_ITERS = 2000

    T.main()

    print(f"\nDone. Per-stage CSVs in {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
