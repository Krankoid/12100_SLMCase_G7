"""
Inference / prompting script (Tiny Shakespeare, char-level).
Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import pickle
import time
import torch
from codecarbon import EmissionsTracker

from model import GPT, GPTConfig

# ----------------------------
# Edit these
# ----------------------------
OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")

PROMPT = "To be, or not to be"
MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0
TOP_K = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sustainability tracking. A single short generation is below CodeCarbon's
# sampling resolution, so we repeat the generation NUM_RUNS times and report
# the per-run average.
RUN_TAG = "baseline"            # change to e.g. "altA", "altB" for other scenarios
NUM_RUNS = 30
# ----------------------------


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # train.py should store config with model parameters and data_dir
    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]

    meta = load_meta(data_dir)
    stoi = meta["stoi"]         # char to index mapping
    itos = meta["itos"]         # index to char mapping

    def encode(s: str):
        # map unknown chars to a safe fallback if needed
        return [stoi.get(ch, stoi[" "]) for ch in s]

    def decode(tokens):
        return "".join([itos[t] for t in tokens])

    config = GPTConfig(**model_cfg)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([encode(PROMPT)], dtype=torch.long, device=DEVICE)

    tracker = EmissionsTracker(
        project_name=f"infer_{RUN_TAG}_tokens{MAX_NEW_TOKENS}_runs{NUM_RUNS}",
        output_dir=OUT_DIR,
        output_file=f"emissions_infer_{RUN_TAG}.csv",
        measure_power_secs=1,
        log_level="warning",
    )
    tracker.start()

    t0 = time.time()
    last_out = None
    for _ in range(NUM_RUNS):
        last_out = model.generate(
            idx,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )
    dt = time.time() - t0

    emissions_kg = tracker.stop()
    energy_kwh = tracker.final_emissions_data.energy_consumed

    print(decode(last_out[0].tolist()))
    print(
        f"\n[CodeCarbon] tag={RUN_TAG} device={DEVICE} tokens={MAX_NEW_TOKENS} "
        f"runs={NUM_RUNS} total_time={dt:.2f}s "
        f"energy_per_run={energy_kwh / NUM_RUNS:.8f} kWh "
        f"emissions_per_run={emissions_kg / NUM_RUNS:.8f} kgCO2e"
    )


if __name__ == "__main__":
    main()
