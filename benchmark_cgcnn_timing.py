#!/usr/bin/env python
"""
Inference Timing Benchmark — CGCNN only
===========================================
Standalone script: run this from your CGCNN virtual environment
(the one where `import cgcnn` works). Does NOT import megnet or
alignn, avoiding cross-library dependency conflicts.

Usage:
    python benchmark_cgcnn_timing.py <modelpath> <cifpath>

Outputs: inference_timing_results/cgcnn_timing.csv
Run benchmark_megnet_timing.py and benchmark_alignn_timing.py
separately (each from their own venv), then run
aggregate_inference_timing.py to combine all three.
"""

import sys
import os
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

OUTPUT_DIR = "inference_timing_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_WARMUP = 5
N_TIMED = 100


def infer_cgcnn_architecture(state_dict):
    """
    Infer CrystalGraphConvNet architecture hyperparameters directly
    from the saved weight tensor shapes, rather than relying on a
    checkpoint['args'] dict that may not be present (checkpoints
    saved by different scripts/tools use different formats).

    Assumes the standard CGCNN parameter naming convention:
        embedding.weight        -> [atom_fea_len, orig_atom_fea_len]
        convs.<i>.fc_full.weight -> [2*atom_fea_len, 2*atom_fea_len + nbr_fea_len]
        conv_to_fc.weight        -> [h_fea_len, atom_fea_len]
        fcs.<i>.weight            -> [h_fea_len, h_fea_len]   (n_h - 1 of these)
    """
    atom_fea_len, orig_atom_fea_len = state_dict["embedding.weight"].shape

    conv_indices = set()
    for key in state_dict:
        if key.startswith("convs.") and key.endswith(".fc_full.weight"):
            conv_indices.add(int(key.split(".")[1]))
    n_conv = max(conv_indices) + 1 if conv_indices else 0

    nbr_fea_len = None
    if n_conv > 0:
        fc_full_in = state_dict["convs.0.fc_full.weight"].shape[1]
        nbr_fea_len = fc_full_in - 2 * atom_fea_len

    h_fea_len = state_dict["conv_to_fc.weight"].shape[0]

    fc_indices = set()
    for key in state_dict:
        if key.startswith("fcs.") and key.endswith(".weight"):
            fc_indices.add(int(key.split(".")[1]))
    n_h = max(fc_indices) + 2 if fc_indices else 1  # +1 for the fcs layers, +1 for conv_to_fc's implicit first hidden layer

    return {
        "orig_atom_fea_len": orig_atom_fea_len,
        "nbr_fea_len": nbr_fea_len,
        "atom_fea_len": atom_fea_len,
        "n_conv": n_conv,
        "h_fea_len": h_fea_len,
        "n_h": n_h,
    }


def main():
    parser = argparse.ArgumentParser(description="CGCNN inference timing benchmark")
    parser.add_argument("modelpath", help="Path to pretrained/fine-tuned CGCNN checkpoint")
    parser.add_argument("cifpath", help="Path to a CIF folder (any element; structure count is what matters)")
    args = parser.parse_args()

    from cgcnn.data import CIFData, collate_pool
    from cgcnn.model import CrystalGraphConvNet

    results = []
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n=== Timing CGCNN ({device_name.upper()}) ===")

        dataset = CIFData(args.cifpath)

        raw = torch.load(args.modelpath, map_location=lambda storage, loc: storage)

        # Handle multiple checkpoint container formats: some save
        # {'state_dict': ..., 'args': ...}, others save the raw
        # state_dict directly, others use different key names.
        if isinstance(raw, dict) and "state_dict" in raw:
            state_dict = raw["state_dict"]
        elif isinstance(raw, dict) and all(
            hasattr(v, "shape") for v in raw.values()
        ):
            # Looks like a raw state_dict (all values are tensors)
            state_dict = raw
        else:
            raise RuntimeError(
                f"Unrecognized checkpoint format in {args.modelpath}. "
                f"Top-level keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}. "
                f"Inspect this checkpoint manually and adjust the loading logic."
            )

        arch = infer_cgcnn_architecture(state_dict)
        print(f"  Inferred architecture: {arch}")

        model = CrystalGraphConvNet(
            arch["orig_atom_fea_len"], arch["nbr_fea_len"],
            atom_fea_len=arch["atom_fea_len"], n_conv=arch["n_conv"],
            h_fea_len=arch["h_fea_len"], n_h=arch["n_h"], classification=False,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=collate_pool
        )

        times = []
        count = 0
        with torch.no_grad():
            for i, (input_data, target, _) in enumerate(loader):
                input_var = tuple(
                    x.to(device) if torch.is_tensor(x) else [t.to(device) for t in x]
                    for x in input_data
                )
                if i < N_WARMUP:
                    model(*input_var)
                    continue
                t0 = time.perf_counter()
                model(*input_var)
                if device_name == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                count += 1
                if count >= N_TIMED:
                    break

        if not times:
            print(f"[SKIP] No timed structures for {device_name}")
            continue

        times_ms = np.array(times) * 1000.0
        mean_ms, std_ms = times_ms.mean(), times_ms.std()
        per_sec = 1000.0 / mean_ms if mean_ms > 0 else float("nan")
        print(f"[CGCNN | {device_name.upper()}] mean={mean_ms:.3f} ms  std={std_ms:.3f} ms  "
              f"({per_sec:.1f} structures/sec)  n={len(times_ms)}")

        results.append({
            "model": "CGCNN", "device": device_name.upper(), "mean_ms": mean_ms,
            "std_ms": std_ms, "structures_per_sec": per_sec, "n_timed": len(times_ms),
        })

    if results:
        result_df = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, "cgcnn_timing.csv")
        result_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
