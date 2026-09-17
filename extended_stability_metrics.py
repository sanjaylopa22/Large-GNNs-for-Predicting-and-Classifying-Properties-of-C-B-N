#!/usr/bin/env python
"""
Extended Stability-Classification Evaluation (Point 5)
==========================================================
Extends the existing stability_classification_{MEGNet,CGCNN,ALIGNN}.py
scripts with:

  1. Matthews Correlation Coefficient (MCC) alongside accuracy/F1,
     per element and pooled (Table: mcc_summary).
  2. A direct comparison between the sign-only criterion (E_f < 0
     alone) and the joint criterion (E_f < 0 AND E_f <= E_hull),
     reporting how many predicted labels differ between the two and
     the resulting change in accuracy/F1/MCC (Table:
     criterion_comparison).
  3. A precision-recall curve (continuous threshold sweep) using
     -E_hat as the ranking score against the joint-criterion ground
     truth label, saved as a plot per model.

Reads the same per-structure stability-label CSVs already produced by
stability_classification_{MEGNet,CGCNN,ALIGNN}.py (columns: true_eform,
pred_eform, e_hull, true_label, pred_label). Update STABILITY_CSV_PATHS
below to point at your actual output files.

Install requirement:
    pip install pandas numpy scikit-learn matplotlib
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, precision_recall_curve, auc,
)

warnings.filterwarnings("ignore")

# ===============================
# USER CONFIG
# ===============================
OUTPUT_DIR = "extended_stability_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Per-structure stability-label CSVs already produced by
# stability_classification_{MEGNet,CGCNN,ALIGNN}.py
STABILITY_CSV_PATHS = {
    "MEGNet": {
        "B": "stability_results_MEGNet/stability_labels_MEGNet_B.csv",
        "C": "stability_results_MEGNet/stability_labels_MEGNet_C.csv",
        "N": "stability_results_MEGNet/stability_labels_MEGNet_N.csv",
    },
    "CGCNN": {
        "B": "stability_results_CGCNN/stability_labels_CGCNN_B.csv",
        "C": "stability_results_CGCNN/stability_labels_CGCNN_C.csv",
        "N": "stability_results_CGCNN/stability_labels_CGCNN_N.csv",
    },
    "ALIGNN": {
        "B": "stability_results_ALIGNN/stability_labels_ALIGNN_B.csv",
        "C": "stability_results_ALIGNN/stability_labels_ALIGNN_C.csv",
        "N": "stability_results_ALIGNN/stability_labels_ALIGNN_N.csv",
    },
}


def classify_sign_only(e_value):
    if pd.isna(e_value):
        return None
    return "Stable" if e_value < 0 else "Unstable"


def load_all_elements(model):
    dfs = []
    for element, path in STABILITY_CSV_PATHS[model].items():
        if not os.path.exists(path):
            print(f"[SKIP] {model} {element}: {path} not found")
            continue
        df = pd.read_csv(path)
        df["element"] = element
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


# ===============================
# PART 1 & 2: MCC + criterion comparison
# ===============================
def mcc_and_criterion_comparison():
    mcc_rows = []
    criterion_rows = []

    for model in STABILITY_CSV_PATHS:
        combined = load_all_elements(model)
        if combined is None:
            continue

        # --- Per-element and combined MCC (joint criterion, as already computed) ---
        per_element_mcc = {}
        for element, sub in combined.groupby("element"):
            mcc = matthews_corrcoef(sub["true_label"], sub["pred_label"])
            per_element_mcc[element] = mcc

        combined_mcc = matthews_corrcoef(combined["true_label"], combined["pred_label"])
        mcc_rows.append({
            "model": model,
            "B_mcc": per_element_mcc.get("B", np.nan),
            "C_mcc": per_element_mcc.get("C", np.nan),
            "N_mcc": per_element_mcc.get("N", np.nan),
            "combined_mcc": combined_mcc,
        })
        print(f"[{model}] MCC — B={per_element_mcc.get('B', float('nan')):.4f}  "
              f"C={per_element_mcc.get('C', float('nan')):.4f}  "
              f"N={per_element_mcc.get('N', float('nan')):.4f}  "
              f"Combined={combined_mcc:.4f}")

        # --- Sign-only vs joint criterion comparison (pooled B+C+N) ---
        combined["true_label_signonly"] = combined["true_eform"].apply(classify_sign_only)
        combined["pred_label_signonly"] = combined["pred_eform"].apply(classify_sign_only)

        valid = combined.dropna(subset=["true_label_signonly", "pred_label_signonly",
                                         "true_label", "pred_label"])

        acc_sign = accuracy_score(valid["true_label_signonly"], valid["pred_label_signonly"])
        f1_sign = f1_score(valid["true_label_signonly"], valid["pred_label_signonly"],
                            pos_label="Stable", zero_division=0)
        mcc_sign = matthews_corrcoef(valid["true_label_signonly"], valid["pred_label_signonly"])

        acc_joint = accuracy_score(valid["true_label"], valid["pred_label"])
        f1_joint = f1_score(valid["true_label"], valid["pred_label"],
                             pos_label="Stable", zero_division=0)
        mcc_joint = matthews_corrcoef(valid["true_label"], valid["pred_label"])

        # How many PREDICTED labels differ between the two criteria
        labels_changed = (valid["pred_label_signonly"] != valid["pred_label"]).sum()
        pct_changed = 100.0 * labels_changed / len(valid)

        criterion_rows.append({
            "model": model,
            "accuracy_signonly": acc_sign, "f1_signonly": f1_sign, "mcc_signonly": mcc_sign,
            "accuracy_joint": acc_joint, "f1_joint": f1_joint, "mcc_joint": mcc_joint,
            "labels_changed": int(labels_changed), "pct_changed": pct_changed,
            "n_total": len(valid),
        })

        print(f"[{model}] Sign-only: Acc={acc_sign:.4f} F1={f1_sign:.4f} MCC={mcc_sign:.4f}")
        print(f"[{model}] Joint    : Acc={acc_joint:.4f} F1={f1_joint:.4f} MCC={mcc_joint:.4f}")
        print(f"[{model}] Predicted labels changed by criterion: {labels_changed} "
              f"({pct_changed:.2f}% of {len(valid)})")

    mcc_df = pd.DataFrame(mcc_rows)
    mcc_path = os.path.join(OUTPUT_DIR, "mcc_summary.csv")
    mcc_df.to_csv(mcc_path, index=False)
    print(f"\nMCC summary saved: {mcc_path}")

    crit_df = pd.DataFrame(criterion_rows)
    crit_path = os.path.join(OUTPUT_DIR, "criterion_comparison.csv")
    crit_df.to_csv(crit_path, index=False)
    print(f"Criterion comparison saved: {crit_path}")

    return mcc_df, crit_df


# ===============================
# PART 3: Precision-recall curves (threshold sweep)
# ===============================
def precision_recall_curves():
    plt.figure(figsize=(7, 6))

    for model in STABILITY_CSV_PATHS:
        combined = load_all_elements(model)
        if combined is None:
            continue

        # Ground truth: joint-criterion true_label (fixed).
        # Score: -pred_eform (higher score = more likely Stable, since
        # lower/more negative formation energy => more stable).
        y_true_binary = (combined["true_label"] == "Stable").astype(int)
        scores = -combined["pred_eform"].astype(float)

        precision, recall, thresholds = precision_recall_curve(y_true_binary, scores)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, label=f"{model} (AUC={pr_auc:.3f})")

        pr_df = pd.DataFrame({"recall": recall, "precision": precision})
        pr_path = os.path.join(OUTPUT_DIR, f"precision_recall_{model}.csv")
        pr_df.to_csv(pr_path, index=False)
        print(f"[{model}] PR-AUC = {pr_auc:.4f}  (curve data saved: {pr_path})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Stability Classification (Stable class)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "precision_recall_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nCombined PR-curve plot saved: {plot_path}")


if __name__ == "__main__":
    mcc_and_criterion_comparison()
    precision_recall_curves()
    print("\nAll outputs written to:", os.path.abspath(OUTPUT_DIR))
