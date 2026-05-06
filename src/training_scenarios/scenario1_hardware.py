"""
Scenario 1 — training hardware sweep.

Sweep DEVICE in {"cpu", "cuda"} with everything else at the shared baseline:
  N_LAYER=4, N_HEAD=4, N_EMBD=128, BATCH_SIZE=32, BLOCK_SIZE=256,
  MAX_ITERS=2000, LEARNING_RATE=3e-4.

The cuda run also produces the baseline checkpoint reused by scenarios 4 and 5
(copied to out_lifecycle/checkpoints/ckpt_baseline.pt).

Run from the project root:
    python -m src.training_scenarios.scenario1_hardware
or:
    cd src && python training_scenarios/scenario1_hardware.py
"""

import importlib
import os
import shutil
import sys

# make `import train_og` work whether we run from project root or src/
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch  # noqa: E402

SCENARIO = "scenario1_hardware"
OUT_DIR = os.path.join("out_lifecycle", SCENARIO)
CKPT_DIR = os.path.join("out_lifecycle", "checkpoints")

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
else:
    print("WARNING: cuda not available, only running cpu sweep value")


def run_one(device: str):
    print(f"\n{'='*60}\nTRAIN [{SCENARIO}] device={device}\n{'='*60}")
    import train_og as T
    importlib.reload(T)

    T.RUN_TAG = device
    T.DEVICE = device
    T.OUT_DIR = OUT_DIR
    # baseline values for everything else
    T.N_LAYER = 4
    T.N_HEAD = 4
    T.N_EMBD = 128
    T.BATCH_SIZE = 32
    T.BLOCK_SIZE = 256
    T.MAX_ITERS = 2000

    T.main()

    # promote the cuda run's checkpoint to baseline so scenarios 4/5 can reuse it
    if device == "cuda":
        os.makedirs(CKPT_DIR, exist_ok=True)
        src_ckpt = os.path.join(OUT_DIR, "ckpt_cuda.pt")
        dst_ckpt = os.path.join(CKPT_DIR, "ckpt_baseline.pt")
        if os.path.exists(src_ckpt):
            shutil.copyfile(src_ckpt, dst_ckpt)
            print(f"[ckpt] copied {src_ckpt} -> {dst_ckpt}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for device in DEVICES:
        run_one(device)
    print(f"\nAll done. Per-stage CSVs in {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
