#!/usr/bin/env python
"""
Inference Timing Benchmark — ALIGNN only
===========================================
Standalone script: run this from your ALIGNN virtual environment
(the one where `import alignn` works, e.g. alignn-main/alignn).
Does NOT import megnet or cgcnn, avoiding cross-library dependency
conflicts.

Reconstructing a fine-tuned ALIGNN model from a saved state_dict
requires the exact ALIGNNConfig used during fine-tuning. Since
fine-tuning only updates weights (the architecture is unchanged from
the pretrained checkpoint), this script pulls the correct config
directly from the pretrained checkpoint's config.json (the same file
your finetuned_alignn_{B,C,N}.py scripts already download via
get_figshare_model()), then loads your fine-tuned weights on top of
that architecture. No guessing required.

Outputs: inference_timing_results/alignn_timing.csv
Run benchmark_megnet_timing.py and benchmark_cgcnn_timing.py
separately (each from their own venv), then run
aggregate_inference_timing.py to combine all three.
"""

import os
import json
import zipfile
import time
import warnings

import numpy as np
import pandas as pd
import requests
import torch

warnings.filterwarnings("ignore")

OUTPUT_DIR = "inference_timing_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_WARMUP = 5
N_TIMED = 100

# ===============================
# USER CONFIG
# ===============================
ALIGNN_MODEL_PATH = "alignn_finetuned_boron.pth"  # UPDATE if needed
ALIGNN_CIF_DIR = "./examples/sample_data/B_materials"  # any element's folder works
CUTOFF = 8.0
MAX_NEIGHBORS = 12

# Same pretrained checkpoint used by finetuned_alignn_{B,C,N}.py to
# derive the model architecture (config.json), matching ALL_MODELS
# in those scripts.
PRETRAINED_MODEL_NAME = "jv_formation_energy_peratom_alignn"
PRETRAINED_MODEL_URL = "https://figshare.com/ndownloader/files/31458679"


def get_pretrained_alignn_config():
    """Download (if needed) the pretrained checkpoint zip and extract
    its config.json, returning the exact ALIGNNConfig kwargs used to
    build the architecture — identical to what your fine-tuning
    scripts used, since fine-tuning does not alter the config."""
    zip_path = PRETRAINED_MODEL_NAME + ".zip"

    if not os.path.exists(zip_path):
        print("Downloading pretrained ALIGNN checkpoint (for its config.json)...")
        r = requests.get(PRETRAINED_MODEL_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as z:
        cfg_file = [x for x in z.namelist() if "config.json" in x][0]
        config = json.loads(z.read(cfg_file))

    return config["model"]


def main():
    from alignn.graphs import Graph
    from alignn.models.alignn import ALIGNN, ALIGNNConfig
    from jarvis.core.atoms import Atoms

    if not os.path.exists(ALIGNN_MODEL_PATH):
        print(f"[FATAL] ALIGNN checkpoint not found: {ALIGNN_MODEL_PATH}")
        return

    print("Extracting ALIGNNConfig from pretrained checkpoint's config.json...")
    config_kwargs = get_pretrained_alignn_config()
    print(f"  Config: {config_kwargs}")

    results = []
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n=== Timing ALIGNN ({device_name.upper()}) ===")

        model = ALIGNN(ALIGNNConfig(**config_kwargs))
        model.load_state_dict(torch.load(ALIGNN_MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        if not os.path.isdir(ALIGNN_CIF_DIR):
            print(f"[FATAL] ALIGNN_CIF_DIR not found: {ALIGNN_CIF_DIR}")
            return

        cif_files = [f for f in os.listdir(ALIGNN_CIF_DIR) if f.endswith(".cif")][:N_WARMUP + N_TIMED]
        if len(cif_files) < N_WARMUP + 1:
            print(f"[FATAL] Not enough .cif files in {ALIGNN_CIF_DIR}")
            return

        times = []
        for i, cif_file in enumerate(cif_files):
            try:
                atoms = Atoms.from_cif(os.path.join(ALIGNN_CIF_DIR, cif_file))
                g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=CUTOFF, max_neighbors=MAX_NEIGHBORS)
                lat = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
                g, lg, lat = g.to(device), lg.to(device), lat.to(device)
            except Exception as e:
                print(f"  [SKIP] {cif_file}: {e}")
                continue

            if i < N_WARMUP:
                with torch.no_grad():
                    model([g, lg, lat])
                continue

            t0 = time.perf_counter()
            with torch.no_grad():
                model([g, lg, lat])
            if device_name == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

            if len(times) >= N_TIMED:
                break

        if not times:
            print(f"[SKIP] No timed structures for {device_name}")
            continue

        times_ms = np.array(times) * 1000.0
        mean_ms, std_ms = times_ms.mean(), times_ms.std()
        per_sec = 1000.0 / mean_ms if mean_ms > 0 else float("nan")
        print(f"[ALIGNN | {device_name.upper()}] mean={mean_ms:.3f} ms  std={std_ms:.3f} ms  "
              f"({per_sec:.1f} structures/sec)  n={len(times_ms)}")

        results.append({
            "model": "ALIGNN", "device": device_name.upper(), "mean_ms": mean_ms,
            "std_ms": std_ms, "structures_per_sec": per_sec, "n_timed": len(times_ms),
        })

    if results:
        result_df = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, "alignn_timing.csv")
        result_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
