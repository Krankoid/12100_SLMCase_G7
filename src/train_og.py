"""
Course training script (simplified from nanoGPT) — instrumented with per-stage
CodeCarbon tracking for the life-cycle assessment in task3/lifecycle_fixes.md.

The training phase is split into six life-cycle stages:
  Part 1 — setup           : load metadata, build model + optimizer
  Part 2 — data_load       : sample a batch from the memmapped dataset
  Part 3 — forward         : forward pass through the model (computes loss)
  Part 4 — backward_update : zero_grad + backward + grad clip + optimizer step
  Part 5 — evaluation      : periodic loss estimation on train/val splits
  Part 6 — checkpoint_save : serialise the model + optimizer to disk

Each stage is wrapped in tracker.start_task("<tag>") / tracker.stop_task().
Per-iteration energies are accumulated into stage_totals and written to a
stages_<RUN_TAG>.csv at the end.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import csv
import time
import pickle
from dataclasses import asdict

import numpy as np
import torch
from codecarbon import EmissionsTracker

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Experiment configuration

# I/O
OUT_DIR = "out_lifecycle/scenario_default"
DATA_DIR = os.path.join("data")
EVAL_INTERVAL = 200
EVAL_ITERS = 50
LOG_INTERVAL = 50
SAVE_CHECKPOINT = True

RUN_TAG = "default"

# Model (main tunables — overridden by per-scenario driver scripts)
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1
BIAS = True

# Training
SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float32"
BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS = 2000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    bin_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, data_dir: str, block_size: int, batch_size: int, device: str, eval_iters: int):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split, data_dir, block_size, batch_size, device)
            _, loss = model(x, y)
            split_losses[k] = loss
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


def save_checkpoint(out_dir: str, model: GPT, optimizer: torch.optim.Optimizer, iter_num: int, config: dict, name: str = "ckpt.pt"):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "iter_num": iter_num,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, os.path.join(out_dir, name))


def _record(stage_totals, stage, emissions_data):
    """Append the (energy_kwh, emissions_kg, duration_s) of one task call."""
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
    set_seed(SEED)

    # one EmissionsTracker for the whole run; per-stage measurement uses
    # tracker.start_task / tracker.stop_task on top of it.
    tracker = EmissionsTracker(
        project_name=f"train_{RUN_TAG}_L{N_LAYER}H{N_HEAD}E{N_EMBD}_{DEVICE}",
        output_dir=OUT_DIR,
        output_file=f"codecarbon_train_{RUN_TAG}.csv",
        measure_power_secs=1,
        log_level="warning",
    )
    tracker.start()

    stage_totals = {
        s: {"energy_kwh": 0.0, "emissions_kg": 0.0, "duration_s": 0.0, "n_calls": 0}
        for s in ["setup", "data_load", "forward", "backward_update", "evaluation", "checkpoint_save"]
    }

    # ------------------------------------------------------------------
    # Part 1 — setup: load metadata, build model + optimizer
    # ------------------------------------------------------------------
    tracker.start_task("setup")

    meta = load_meta(DATA_DIR)
    vocab_size = meta["vocab_size"] if meta and "vocab_size" in meta else 50304

    cfg = GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS,
    )

    model = GPT(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    _record(stage_totals, "setup", tracker.stop_task())

    t0 = time.time()
    for it in range(MAX_ITERS + 1):

        if it % EVAL_INTERVAL == 0:
            # ----------------------------------------------------------
            # Part 5 — evaluation: estimate train/val loss
            # ----------------------------------------------------------
            tracker.start_task("evaluation")
            losses = estimate_loss(model, DATA_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS)
            _record(stage_totals, "evaluation", tracker.stop_task())

            dt = time.time() - t0
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | elapsed {dt:.1f}s")

            if SAVE_CHECKPOINT and it > 0:
                # ------------------------------------------------------
                # Part 6 — checkpoint_save: serialise model + optimizer
                # ------------------------------------------------------
                tracker.start_task("checkpoint_save")
                config_dump = {
                    "data_dir": DATA_DIR,
                    "train": {
                        "batch_size": BATCH_SIZE, "block_size": BLOCK_SIZE,
                        "max_iters": MAX_ITERS, "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                        "dtype": DTYPE, "device": DEVICE,
                    },
                    "model": asdict(cfg),
                }
                save_checkpoint(OUT_DIR, model, optimizer, it, config_dump)
                _record(stage_totals, "checkpoint_save", tracker.stop_task())

        # --------------------------------------------------------------
        # Part 2 — data_load: sample a batch from the memmap dataset
        # --------------------------------------------------------------
        tracker.start_task("data_load")
        x, y = get_batch("train", DATA_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _record(stage_totals, "data_load", tracker.stop_task())

        # --------------------------------------------------------------
        # Part 3 — forward: forward pass through the model (loss)
        # --------------------------------------------------------------
        tracker.start_task("forward")
        _, loss = model(x, y)
        _record(stage_totals, "forward", tracker.stop_task())

        # --------------------------------------------------------------
        # Part 4 — backward_update: zero_grad + backward + clip + step
        # --------------------------------------------------------------
        tracker.start_task("backward_update")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        _record(stage_totals, "backward_update", tracker.stop_task())

        if it % LOG_INTERVAL == 0:
            print(f"iter {it:5d} | loss {loss.item():.4f}")

    # final checkpoint (also instrumented as Part 6)
    if SAVE_CHECKPOINT:
        tracker.start_task("checkpoint_save")
        config_dump = {
            "data_dir": DATA_DIR,
            "train": {
                "batch_size": BATCH_SIZE, "block_size": BLOCK_SIZE,
                "max_iters": MAX_ITERS, "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                "dtype": DTYPE, "device": DEVICE,
            },
            "model": asdict(cfg),
        }
        save_checkpoint(OUT_DIR, model, optimizer, MAX_ITERS, config_dump,
                        name=f"ckpt_{RUN_TAG}.pt")
        _record(stage_totals, "checkpoint_save", tracker.stop_task())

    tracker.stop()
    _write_stages_csv(OUT_DIR, RUN_TAG, stage_totals)

    total_kwh = sum(d["energy_kwh"] for d in stage_totals.values())
    total_kg = sum(d["emissions_kg"] for d in stage_totals.values())
    print(f"[total] tag={RUN_TAG} device={DEVICE} energy={total_kwh:.6f} kWh "
          f"emissions={total_kg:.6f} kgCO2e")


if __name__ == "__main__":
    main()
