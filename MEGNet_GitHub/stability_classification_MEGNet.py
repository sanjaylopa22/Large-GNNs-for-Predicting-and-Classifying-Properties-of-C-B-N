#!/usr/bin/env python
"""
Module 7 — Thermodynamic Stability Classification (MEGNet)
============================================================
For each of the B / C / N elemental subsets, this script:

  1. Loads the MEGNet prediction CSVs already produced by
     Megnet_Finetune_random_{B,C,N}.py (columns: id, true_eform, pred_eform).
  2. Fetches energy-above-hull (E_hull) values for the corresponding
     Materials Project IDs via the Materials Project API (mp-api).
  3. Classifies each structure as Stable / Unstable using BOTH the
     predicted formation energy (Ê_f) and the ground-truth formation
     energy (E_f), applying the joint criterion:

        Stable  <=>  E < 0  AND  E <= E_hull
        Unstable<=>  E > 0  OR   E >  E_hull

  4. Treats the ground-truth-derived label as the reference class and
     the prediction-derived label as the "classifier" output, then
     computes Accuracy, Precision, Recall, and F1-score (positive
     class = "Stable") per element and combined (micro-pooled).
  5. Saves per-structure labeled CSVs, a confusion-matrix plot, a
     stability-distribution bar plot, and a summary metrics table.

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

ELEMENTS = ["B", "C", "N"]

# Prediction CSVs produced by Megnet_Finetune_random_{elem}.py
# (columns expected: id, true_eform, pred_eform)
PRED_CSV_PATHS = {
    "B": "megnet_results_B_finetune.csv",
    "C": "megnet_results_C_finetune.csv",
    "N": "megnet_results_N_finetune.csv",
}

OUTPUT_DIR = "stability_results_MEGNet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 1000  # material_ids per MP API query batch


# ===============================
# HELPER: normalize material IDs
# ===============================
def normalize_mp_id(raw_id):
    """Ensure IDs look like 'mp-1234' (MEGNet CSVs may store bare ids)."""
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
    """
    Query the Materials Project API for energy_above_hull for a list
    of material IDs. Returns dict: {material_id: e_hull}.
    """
    from mp_api.client import MPRester

    ehull_map = {}
    ids = list(dict.fromkeys(material_ids))  # de-dup, preserve order
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
            time.sleep(0.2)  # be gentle with the API

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
        print(f" Element: {elem}")
        print("=" * 60)

        csv_path = PRED_CSV_PATHS[elem]
        if not os.path.exists(csv_path):
            print(f"  [SKIP] Prediction file not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        required = {"id", "true_eform", "pred_eform"}
        if not required.issubset(df.columns):
            raise ValueError(f"{csv_path} must contain columns: {required}")

        df["material_id"] = df["id"].apply(normalize_mp_id)

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
        out_csv = os.path.join(OUTPUT_DIR, f"stability_labels_MEGNet_{elem}.csv")
        df.to_csv(out_csv, index=False)
        print(f"  Saved per-structure labels: {out_csv}")

        # --- Metrics (positive class = Stable) ---
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
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Stable", "Unstable"])
        ax.set_yticklabels(["Stable", "Unstable"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"MEGNet — {elem} Stability Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black", fontsize=12)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_MEGNet_{elem}.png")
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
        ax.set_title(f"MEGNet — {elem} Stability Distribution")
        ax.legend()
        fig.tight_layout()
        dist_path = os.path.join(OUTPUT_DIR, f"stability_distribution_MEGNet_{elem}.png")
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
    summary_path = os.path.join(OUTPUT_DIR, "summary_metrics_MEGNet_stability.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(" SUMMARY — MEGNet Stability Classification")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    print("All outputs written to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
