#!/usr/bin/env python
"""
Error Diagnostics (Point 4)
=============================
1. Residual plots (E_hat - E_true vs E_true) per element/model, to
   check for systematic bias as a function of formation-energy
   magnitude, plus the Pearson correlation between |residual| and
   |E_true| (Table: error_magnitude_correlation).
2. Carbon-subset MAE stratified by mean atomic coordination number,
   used as a structural proxy for hybridization (sp3 ~ 4-fold
   coordination, sp2 ~ 3-fold, sp ~ 2-fold), computed directly from
   the CIF structures via pymatgen's CrystalNN local-environment
   analyzer (Table: carbon_hybridization_error).

Reads prediction CSVs already produced by your dataset-specific
fine-tuning scripts. Update PRED_CSV_PATHS and CIF_DIRS below to
match your actual file locations and column conventions per model.

Install requirement:
    pip install pandas numpy scipy matplotlib pymatgen
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

warnings.filterwarnings("ignore")

# ===============================
# USER CONFIG
# ===============================
OUTPUT_DIR = "error_diagnostics_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prediction CSVs (dataset-specific fine-tuned results), with the
# column-name convention each model actually uses.
PRED_CSV_CONFIG = {
    "MEGNet": {
        "B": {"path": "megnet_results_B_finetune.csv", "id_col": "id", "true_col": "true_eform", "pred_col": "pred_eform"},
        "C": {"path": "megnet_results_C_finetune.csv", "id_col": "id", "true_col": "true_eform", "pred_col": "pred_eform"},
        "N": {"path": "megnet_results_N_finetune.csv", "id_col": "id", "true_col": "true_eform", "pred_col": "pred_eform"},
    },
    "CGCNN": {
        "B": {"path": "test_results.csv", "id_col": "Material", "true_col": "Target_FE", "pred_col": "Predicted_FE"},
        "C": {"path": "test_results_Carbon.csv", "id_col": "Material", "true_col": "Target_FE", "pred_col": "Predicted_FE"},
        "N": {"path": "test_results_nitrogen_materials.csv", "id_col": "Material", "true_col": "Target_FE", "pred_col": "Predicted_FE"},
    },
    "ALIGNN": {
        "B": {"path": "boron_test_predictions.csv", "id_col": "file_name", "true_col": "formation_energy_per_atom", "pred_col": "predicted_energy_per_atom"},
        "C": {"path": "carbon_test_predictions.csv", "id_col": "file_name", "true_col": "formation_energy_per_atom", "pred_col": "predicted_energy_per_atom"},
        "N": {"path": "nitrogen_test_predictions.csv", "id_col": "file_name", "true_col": "formation_energy_per_atom", "pred_col": "predicted_energy_per_atom"},
    },
}

# CIF directory for Carbon, needed for coordination-number analysis
CARBON_CIF_DIR = "test_cif/C_materials"


# ===============================
# PART 1: Residual plots + error-magnitude correlation
# ===============================
def residual_analysis():
    correlation_rows = []

    for model, elements in PRED_CSV_CONFIG.items():
        for element, cfg in elements.items():
            if not os.path.exists(cfg["path"]):
                print(f"[SKIP] {model} {element}: {cfg['path']} not found")
                continue

            df = pd.read_csv(cfg["path"])
            true_vals = df[cfg["true_col"]].astype(float)
            pred_vals = df[cfg["pred_col"]].astype(float)
            residual = pred_vals - true_vals

            # Residual plot
            plt.figure(figsize=(6, 5))
            plt.scatter(true_vals, residual, alpha=0.6, s=20, edgecolors="k", linewidths=0.3)
            plt.axhline(0, color="red", linestyle="--", linewidth=1.2)
            plt.xlabel("True Formation Energy (eV/atom)")
            plt.ylabel("Residual (Predicted - True, eV/atom)")
            plt.title(f"{model} — {element} Residuals vs. True $E_f$")
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, f"residuals_{model}_{element}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"[{model} {element}] Residual plot saved: {plot_path}")

            # Correlation between |residual| and |E_true|
            abs_residual = np.abs(residual)
            abs_true = np.abs(true_vals)
            if len(abs_residual) > 2:
                r, p = pearsonr(abs_true, abs_residual)
            else:
                r, p = np.nan, np.nan

            correlation_rows.append({
                "model": model, "element": element,
                "pearson_r": r, "p_value": p, "n": len(df),
            })
            print(f"[{model} {element}] Pearson r(|E_true|, |residual|) = {r:.4f} (p={p:.4g})")

    corr_df = pd.DataFrame(correlation_rows)
    corr_path = os.path.join(OUTPUT_DIR, "error_magnitude_correlation.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"\nCorrelation summary saved: {corr_path}")
    return corr_df


# ===============================
# PART 2: Carbon hybridization-stratified error
# ===============================
def estimate_mean_coordination(cif_path):
    """Return the mean coordination number of carbon atoms in a structure,
    using CrystalNN. Returns None if the structure fails to parse."""
    try:
        structure = Structure.from_file(cif_path)
        cnn = CrystalNN()
        coord_numbers = []
        for i, site in enumerate(structure):
            if site.specie.symbol != "C":
                continue
            try:
                cn = cnn.get_cn(structure, i)
                coord_numbers.append(cn)
            except Exception:
                continue
        if not coord_numbers:
            return None
        return float(np.mean(coord_numbers))
    except Exception as e:
        print(f"  [ERROR] {cif_path}: {e}")
        return None


def hybridization_bucket(mean_cn):
    if mean_cn is None:
        return None
    if mean_cn >= 3.5:
        return "sp3-like"
    elif mean_cn >= 2.5:
        return "sp2-like"
    else:
        return "sp-like"


def carbon_hybridization_analysis():
    print("\n=== Carbon hybridization-stratified error analysis ===")
    summary_rows = []

    for model, elements in PRED_CSV_CONFIG.items():
        cfg = elements.get("C")
        if cfg is None or not os.path.exists(cfg["path"]):
            print(f"[SKIP] {model}: Carbon predictions not found")
            continue

        df = pd.read_csv(cfg["path"])
        df["material_id"] = df[cfg["id_col"]].astype(str).str.replace(".cif", "", regex=False)
        df["true_eform"] = df[cfg["true_col"]].astype(float)
        df["pred_eform"] = df[cfg["pred_col"]].astype(float)
        df["abs_error"] = (df["pred_eform"] - df["true_eform"]).abs()

        # Compute mean coordination number per structure
        mean_cns = []
        for material_id in df["material_id"]:
            cif_file = material_id if material_id.endswith(".cif") else f"{material_id}.cif"
            cif_path = os.path.join(CARBON_CIF_DIR, cif_file)
            if os.path.exists(cif_path):
                mean_cns.append(estimate_mean_coordination(cif_path))
            else:
                mean_cns.append(None)

        df["mean_cn"] = mean_cns
        df["hybridization"] = df["mean_cn"].apply(hybridization_bucket)
        df = df.dropna(subset=["hybridization"])

        if df.empty:
            print(f"[SKIP] {model}: no structures with resolvable coordination number")
            continue

        detail_path = os.path.join(OUTPUT_DIR, f"carbon_hybridization_detail_{model}.csv")
        df.to_csv(detail_path, index=False)
        print(f"[{model}] Per-structure hybridization detail saved: {detail_path}")

        for hyb in ["sp3-like", "sp2-like", "sp-like"]:
            subset = df[df["hybridization"] == hyb]
            if len(subset) == 0:
                mae, n = np.nan, 0
            else:
                mae, n = subset["abs_error"].mean(), len(subset)
            summary_rows.append({"model": model, "hybridization": hyb, "mae": mae, "n": n})
            print(f"  [{model}] {hyb}: MAE={mae if not np.isnan(mae) else 'N/A'}  (n={n})")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "carbon_hybridization_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCarbon hybridization summary saved: {summary_path}")
    return summary_df


if __name__ == "__main__":
    residual_analysis()
    carbon_hybridization_analysis()
    print("\nAll diagnostic outputs written to:", os.path.abspath(OUTPUT_DIR))
