#!/usr/bin/env python
"""
Module 7 — Thermodynamic Stability Classification (ALIGNN, UNFROZEN-BACKBONE ABLATION)
==========================================================================================
This is the unfrozen-backbone variant of stability_classification_ALIGNN.py.
It runs the same Module 7 stability-classification pipeline, but on the
predictions produced by the protocol-controlled ablation
(alignn_unfrozen_backbone_ablation.py) rather than the original
frozen-backbone fine-tuning (finetuned_alignn_{B,C,N}.py). This allows
direct comparison of stability-classification reliability between
ALIGNN's original (frozen-backbone) and ablated (unfrozen-backbone)
configurations, extending the regression-only ablation
(Table~\\ref{tab:protocol_ablation}) into the downstream classification
task.

For each of the B / C / N elemental subsets, this script:

  1. Loads the ALIGNN unfrozen-backbone ablation prediction CSVs
     produced by alignn_unfrozen_backbone_ablation.py
     (columns: file_name, predicted_energy_per_atom, formation_energy_per_atom).
  2. Fetches energy-above-hull (E_hull) values for the corresponding
     Materials Project IDs via the Materials Project API (mp-api).
  3. Classifies each structure as Stable / Unstable using BOTH the
     predicted formation energy (Ê_f) and the ground-truth formation
     energy (E_f), applying the joint criterion:

        Stable  <=>  E < 0  AND  E <= E_hull
        Unstable<=>  E > 0  OR   E >  E_hull

  4. Computes Accuracy, Precision, Recall, F1-score (positive class =
     "Stable") comparing the prediction-derived label against the
     ground-truth-derived label, per element and combined.
  5. Saves per-structure labeled CSVs, confusion-matrix plots,
     stability-distribution plots, and a summary metrics table, all
     under a SEPARATE output directory from the original
     (frozen-backbone) run, so both result sets are preserved for
     side-by-side comparison.

NOTE ON MATERIAL IDs:
  The ALIGNN "file_name" column is the CIF filename (e.g. "mp-1234.cif").
  This script assumes CIF files are named after their Materials Project
  ID. If your CIF naming convention differs, edit `normalize_mp_id()`.

Install requirement:
    pip install mp-api pandas numpy matplotlib scikit-learn
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ===============================
# USER CONFIG
# ===============================
MP_API_KEY = os.environ.get("MP_API_KEY", "jg5YFc4i3f7KmWdZNCALVB97QIgZTji5")
# NOTE: a real API key was previously hardcoded here as a fallback default
# in an earlier version of this script. It has been removed. Rotate that
# key on the Materials Project dashboard if it was ever committed/shared,
# and set MP_API_KEY as an environment variable instead of hardcoding a
# key in this file.

ELEMENTS = ["B", "C", "N"]

# Prediction CSVs produced by alignn_unfrozen_backbone_ablation.py
# (columns expected: file_name, predicted_energy_per_atom, formation_energy_per_atom)
PRED_CSV_PATHS = {
    "B": "output_ALIGNN_unfrozen_ablation/B/predictions_B_unfrozen.csv",
    "C": "output_ALIGNN_unfrozen_ablation/C/predictions_C_unfrozen.csv",
    "N": "output_ALIGNN_unfrozen_ablation/N/predictions_N_unfrozen.csv",
}

# Separate output directory from the original frozen-backbone stability
# results (stability_results_ALIGNN/), so both sets are preserved for
# side-by-side comparison in the paper.
OUTPUT_DIR = "stability_results_ALIGNN_unfrozen"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 1000  # material_ids per MP API query batch


# ===============================
# HELPER: normalize material IDs
# ===============================
def normalize_mp_id(raw_id):
    """Ensure IDs look like 'mp-1234' (ALIGNN 'file_name' col = CIF stem)."""
    s = str(raw_id).strip().replace(".cif", "")
    if s.lower().startswith("mp-"):
        return s.lower()
    if s.isdigit():
        return f"mp-{s}"
    return s


# ===============================
# HELPER: fetch E_hull from Materials Project
# ===============================
def fetch_ehull(material_ids):
    from mp_api.client import MPRester

    ehull_map = {}
    ids = list(dict.fromkeys(material_ids))
    print(f"  Querying Materials Project for {len(ids)} material IDs...")

    with MPRester(MP_API_KEY) as mpr:
        for start in range(0, len(ids), BATCH_SIZE):
            chunk = ids[start:start + BATCH_SIZE]
            try:
                docs = mpr.materials.summary.search(
                    material_ids=chunk,
                    fields=["material_id", "energy_above_hull"],
                )
                for d in docs:
                    ehull_map[str(d.material_id)] = float(d.energy_above_hull)
            except Exception as e:
                print(f"  [WARNING] MP query failed for batch starting at {start}: {e}")
            time.sleep(0.2)

    print(f"  Retrieved E_hull for {len(ehull_map)} / {len(ids)} materials.")
    return ehull_map


# ===============================
# HELPER: joint stability criterion
# ===============================
def classify_stability(e_value, e_hull):
    if pd.isna(e_value) or pd.isna(e_hull):
        return None
    if e_value < 0 and e_value <= e_hull:
        return "Stable"
    return "Unstable"


# ===============================
# MAIN
# ===============================
def main():
    all_true_labels = []
    all_pred_labels = []
    summary_rows = []

    for elem in ELEMENTS:
        print("\n" + "=" * 60)
        print(f" Element: {elem}  (ALIGNN, unfrozen backbone)")
        print("=" * 60)

        csv_path = PRED_CSV_PATHS[elem]
        if not os.path.exists(csv_path):
            print(f"  [SKIP] Prediction file not found: {csv_path}")
            print(f"         Run alignn_unfrozen_backbone_ablation.py first.")
            continue

        df = pd.read_csv(csv_path)
        required = {"file_name", "predicted_energy_per_atom", "formation_energy_per_atom"}
        if not required.issubset(df.columns):
            raise ValueError(f"{csv_path} must contain columns: {required}")

        df = df.rename(columns={
            "predicted_energy_per_atom": "pred_eform",
            "formation_energy_per_atom": "true_eform",
        })
        df["material_id"] = df["file_name"].apply(normalize_mp_id)

        # --- Fetch E_hull ---
        ehull_map = fetch_ehull(df["material_id"].tolist())
        df["e_hull"] = df["material_id"].map(ehull_map)

        n_before = len(df)
        df = df.dropna(subset=["e_hull"]).reset_index(drop=True)
        print(f"  Matched {len(df)} / {n_before} structures with E_hull data.")

        if df.empty:
            print(f"  [SKIP] No structures with valid E_hull for element {elem}.")
            continue

        # --- Classify (predicted vs ground truth) ---
        df["true_label"] = df.apply(
            lambda r: classify_stability(r["true_eform"], r["e_hull"]), axis=1
        )
        df["pred_label"] = df.apply(
            lambda r: classify_stability(r["pred_eform"], r["e_hull"]), axis=1
        )

        df = df.dropna(subset=["true_label", "pred_label"]).reset_index(drop=True)

        # --- Save per-structure labeled CSV ---
        out_csv = os.path.join(OUTPUT_DIR, f"stability_labels_ALIGNN_unfrozen_{elem}.csv")
        df.to_csv(out_csv, index=False)
        print(f"  Saved per-structure labels: {out_csv}")

        # --- Metrics ---
        y_true = df["true_label"].values
        y_pred = df["pred_label"].values

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label="Stable", zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label="Stable", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label="Stable", zero_division=0)

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-score  : {f1:.4f}")

        summary_rows.append({
            "Element": elem,
            "N_structures": len(df),
            "N_true_stable": int((y_true == "Stable").sum()),
            "N_pred_stable": int((y_pred == "Stable").sum()),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_score": round(f1, 4),
        })

        all_true_labels.extend(y_true.tolist())
        all_pred_labels.extend(y_pred.tolist())

        # --- Confusion matrix plot ---
        cm = confusion_matrix(y_true, y_pred, labels=["Stable", "Unstable"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Stable", "Unstable"])
        ax.set_yticklabels(["Stable", "Unstable"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"ALIGNN (Unfrozen Backbone) — {elem} Stability Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black", fontsize=12)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_ALIGNN_unfrozen_{elem}.png")
        fig.savefig(cm_path, dpi=300)
        plt.close(fig)
        print(f"  Confusion matrix saved: {cm_path}")

        # --- Stability distribution bar plot ---
        fig, ax = plt.subplots(figsize=(5, 4))
        counts_true = [int((y_true == "Stable").sum()), int((y_true == "Unstable").sum())]
        counts_pred = [int((y_pred == "Stable").sum()), int((y_pred == "Unstable").sum())]
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width / 2, counts_true, width, label="Ground Truth")
        ax.bar(x + width / 2, counts_pred, width, label="Predicted")
        ax.set_xticks(x)
        ax.set_xticklabels(["Stable", "Unstable"])
        ax.set_ylabel("Number of structures")
        ax.set_title(f"ALIGNN (Unfrozen Backbone) — {elem} Stability Distribution")
        ax.legend()
        fig.tight_layout()
        dist_path = os.path.join(OUTPUT_DIR, f"stability_distribution_ALIGNN_unfrozen_{elem}.png")
        fig.savefig(dist_path, dpi=300)
        plt.close(fig)
        print(f"  Distribution plot saved: {dist_path}")

    # ===============================
    # COMBINED (MICRO-POOLED) METRICS
    # ===============================
    if all_true_labels:
        acc = accuracy_score(all_true_labels, all_pred_labels)
        prec = precision_score(all_true_labels, all_pred_labels, pos_label="Stable", zero_division=0)
        rec = recall_score(all_true_labels, all_pred_labels, pos_label="Stable", zero_division=0)
        f1 = f1_score(all_true_labels, all_pred_labels, pos_label="Stable", zero_division=0)

        summary_rows.append({
            "Element": "Combined (B+C+N)",
            "N_structures": len(all_true_labels),
            "N_true_stable": int(sum(l == "Stable" for l in all_true_labels)),
            "N_pred_stable": int(sum(l == "Stable" for l in all_pred_labels)),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_score": round(f1, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "summary_metrics_ALIGNN_unfrozen_stability.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(" SUMMARY — ALIGNN (Unfrozen Backbone) Stability Classification")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    print("All outputs written to:", os.path.abspath(OUTPUT_DIR))
    print("\nCompare against stability_results_ALIGNN/summary_metrics_ALIGNN_stability.csv "
          "(the original frozen-backbone results) to complete the ablation's "
          "downstream stability-classification comparison.")


if __name__ == "__main__":
    main()
