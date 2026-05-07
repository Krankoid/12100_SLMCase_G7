"""
Scenario 1 — GPU half only.

Trains the baseline configuration (N_LAYER=4, N_HEAD=4, N_EMBD=128,
BATCH_SIZE=32, BLOCK_SIZE=256, MAX_ITERS=2000) on cuda and promotes the
resulting checkpoint to ckpt_baseline.pt so scenarios 4 and 5 can reuse it.

Outputs go to out_lifecycle/scenario1_hardware/ — same folder as the cpu
half, so plotting code that reads that folder will see both runs once both
scripts have completed.

Run:
    python -m src.training_scenarios.scenario1_hardware_gpu
"""

import importlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch  # noqa: E402

SCENARIO = "scenario1_hardware"
OUT_DIR = os.path.join("out_lifecycle", SCENARIO)
CKPT_DIR = os.path.join("out_lifecycle", "checkpoints")
DEVICE = "cuda"


def main():
    if not torch.cuda.is_available():
        print("ERROR: cuda not available. Cannot run the GPU half of scenario 1.")
        sys.exit(1)

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

    os.makedirs(CKPT_DIR, exist_ok=True)
    src_ckpt = os.path.join(OUT_DIR, "ckpt_cuda.pt")
    dst_ckpt = os.path.join(CKPT_DIR, "ckpt_baseline.pt")
    if os.path.exists(src_ckpt):
        shutil.copyfile(src_ckpt, dst_ckpt)
        print(f"[ckpt] copied {src_ckpt} -> {dst_ckpt}")

    print(f"\nDone. Per-stage CSVs in {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
