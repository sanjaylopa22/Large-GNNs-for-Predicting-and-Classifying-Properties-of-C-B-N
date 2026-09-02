import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pymatgen.core import Structure
from megnet.utils.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ===============================
# USER CONFIG
# ===============================
BASE_DIR      = "test_cif"                        # root folder containing element subfolders
ELEMENTS      = ["B", "C", "N"]                   # elements to process
OUTPUT_DIR    = "megnet_results_BCN_hybrid"              # folder for all output CSVs and plots
SEED          = 42
EPOCHS        = 50
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# HELPER: Build paths per element
# ===============================
def get_element_paths(element):
    cif_dir  = os.path.join(BASE_DIR, f"{element}_materials")
    csv_path = os.path.join(cif_dir, f"id_prop_{element}.csv")
    return cif_dir, csv_path

# ===============================
# HELPER: Load structures safely
# ===============================
def load_structures_and_targets(subset_df, cif_dir, label=""):
    structures, targets, valid_ids = [], [], []
    missing, errors = 0, 0

    for _, row in subset_df.iterrows():
        material_id = str(row["id"])
        true_eform  = float(row["formation_energy"])
        cif_file    = material_id if material_id.endswith(".cif") else f"{material_id}.cif"
        cif_path    = os.path.join(cif_dir, cif_file)

        if not os.path.exists(cif_path):
            print(f"  [WARNING] Missing CIF ({label}): {cif_path}")
            missing += 1
            continue
        try:
            structure = Structure.from_file(cif_path)
            structures.append(structure)
            targets.append(true_eform)
            valid_ids.append(material_id)
        except Exception as e:
            print(f"  [ERROR] ({label}) {material_id} -> {e}")
            errors += 1

    print(f"  {label}: loaded {len(structures)} / {len(subset_df)}  "
          f"(missing={missing}, errors={errors})")
    return structures, targets, valid_ids

# ============================================================
# STEP 1: Hybrid 80/20 split per element, collect train pool
# ============================================================
print("=" * 60)
print(" STEP 1: Per-element 80/20 split (seed=42)")
print("=" * 60)

per_element_data = {}   # stores split info keyed by element
all_train_structures = []
all_train_targets    = []

for elem in ELEMENTS:
    cif_dir, csv_path = get_element_paths(elem)
    print(f"\n[{elem}] Reading CSV: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"id", "formation_energy"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"[{elem}] CSV must have columns: {required_cols}")

    print(f"  Total samples: {len(df)}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # Load training structures for this element
    print(f"  Loading TRAIN structures for [{elem}]...")
    tr_structs, tr_targets, tr_ids = load_structures_and_targets(
        train_df, cif_dir, label=f"{elem}-TRAIN"
    )

    # Load test structures for this element (kept separate for per-element eval)
    print(f"  Loading TEST structures for [{elem}]...")
    te_structs, te_targets, te_ids = load_structures_and_targets(
        test_df, cif_dir, label=f"{elem}-TEST"
    )

    per_element_data[elem] = {
        "cif_dir"     : cif_dir,
        "test_structs": te_structs,
        "test_targets": te_targets,
        "test_ids"    : te_ids,
    }

    # Accumulate into combined training pool
    all_train_structures.extend(tr_structs)
    all_train_targets.extend(tr_targets)

print(f"\nCombined training pool: {len(all_train_structures)} structures")

# ============================================================
# STEP 2: Load pretrained MEGNet, recompile for fine-tuning
# ============================================================
print("\n" + "=" * 60)
print(" STEP 2: Load pretrained MEGNet & recompile")
print("=" * 60)

import tensorflow as tf

print("Loading pretrained MEGNet (Eform_MP_2018)...")
model = load_model("Eform_MP_2018")

model.model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mae"
)
print(f"  Optimizer : Adam  |  LR = {LEARNING_RATE}  |  Loss = MAE")

# ============================================================
# STEP 3: Fine-tune on combined B+C+N training set
# ============================================================
print("\n" + "=" * 60)
print(f" STEP 3: Fine-tuning on combined B+C+N training set")
print(f"         Epochs={EPOCHS}, BatchSize={BATCH_SIZE}")
print("=" * 60)

history = model.train(
    all_train_structures,
    all_train_targets,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    scrub_failed_structures=True,
)
print("\nFine-tuning complete.\n")

# Save training loss curve
if history is not None:
    try:
        loss_key = [k for k in history.history.keys() if "loss" in k.lower()][0]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.history[loss_key], lw=2, label="Train Loss (MAE)")
        if "val_" + loss_key in history.history:
            ax.plot(history.history["val_" + loss_key], lw=2, label="Val Loss (MAE)", linestyle="--")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MAE Loss (eV/atom)", fontsize=12)
        ax.set_title("MEGNet Fine-Tuning Loss — Combined B+C+N Training Set", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        loss_path = os.path.join(OUTPUT_DIR, "finetune_loss_curve_BCN.png")
        fig.savefig(loss_path, dpi=300)
        plt.close(fig)
        print(f"Loss curve saved: {loss_path}")
    except Exception as e:
        print(f"[INFO] Could not plot loss curve: {e}")

# ============================================================
# STEP 4: Per-element prediction, CSV export, metrics & plots
# ============================================================
print("\n" + "=" * 60)
print(" STEP 4: Per-element test evaluation")
print("=" * 60)

summary_rows = []

# Color palette for each element
elem_colors = {"B": "#E05C2A", "C": "#2A7EC0", "N": "#2AB87A"}

# Collect all results for the combined parity plot
all_results = []

for elem in ELEMENTS:
    data = per_element_data[elem]
    test_structs  = data["test_structs"]
    test_targets  = data["test_targets"]
    test_ids      = data["test_ids"]

    print(f"\n[{elem}] Predicting {len(test_structs)} test structures...")

    results = []
    for structure, true_eform, material_id in zip(test_structs, test_targets, test_ids):
        try:
            pred_eform = float(model.predict_structure(structure).ravel()[0])
            results.append({
                "element"   : elem,
                "id"        : material_id,
                "true_eform": true_eform,
                "pred_eform": pred_eform,
                "error"     : pred_eform - true_eform,
                "abs_error" : abs(pred_eform - true_eform),
            })
        except Exception as e:
            print(f"  [ERROR] {material_id} -> {e}")

    if len(results) == 0:
        print(f"  [SKIP] No predictions for element {elem}.")
        continue

    results_df = pd.DataFrame(results)
    all_results.append(results_df)

    # --- Save per-element CSV ---
    csv_out = os.path.join(OUTPUT_DIR, f"predictions_{elem}.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # --- Metrics ---
    mae  = mean_absolute_error(results_df["true_eform"], results_df["pred_eform"])
    mse  = mean_squared_error(results_df["true_eform"], results_df["pred_eform"])
    rmse = np.sqrt(mse)

    print(f"  [{elem}] Test Samples : {len(results_df)}")
    print(f"  [{elem}] MAE          : {mae:.4f}  eV/atom")
    print(f"  [{elem}] MSE          : {mse:.4f}  (eV/atom)²")
    print(f"  [{elem}] RMSE         : {rmse:.4f}  eV/atom")

    summary_rows.append({
        "Element"     : elem,
        "Test_Samples": len(results_df),
        "MAE"         : round(mae, 4),
        "MSE"         : round(mse, 4),
        "RMSE"        : round(rmse, 4),
    })

    # --- Per-element parity scatter plot ---
    color = elem_colors.get(elem, "#555555")
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        results_df["true_eform"],
        results_df["pred_eform"],
        alpha=0.75,
        color=color,
        edgecolors="k",
        linewidths=0.4,
        s=50,
        zorder=3,
    )

    min_val = min(results_df["true_eform"].min(), results_df["pred_eform"].min())
    max_val = max(results_df["true_eform"].max(), results_df["pred_eform"].max())
    margin  = 0.05 * (max_val - min_val)
    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            "r--", lw=1.5, label="Ideal (y = x)", zorder=2)

    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    ax.set_xlabel("True Formation Energy (eV/atom)", fontsize=12)
    ax.set_ylabel("Predicted Formation Energy (eV/atom)", fontsize=12)
    ax.set_title(
        f"MEGNet Fine-Tuned — {elem} Test Set\n"
        f"MAE = {mae:.4f}  |  MSE = {mse:.4f}  |  RMSE = {rmse:.4f}  eV/atom",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    plot_out = os.path.join(OUTPUT_DIR, f"parity_plot_{elem}.png")
    fig.savefig(plot_out, dpi=300)
    plt.close(fig)
    print(f"  Parity plot saved: {plot_out}")

# ============================================================
# STEP 5: Combined 3-panel parity plot (B | C | N side-by-side)
# ============================================================
if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    for idx, elem in enumerate(ELEMENTS):
        sub = combined_df[combined_df["element"] == elem]
        if sub.empty:
            continue

        mae  = mean_absolute_error(sub["true_eform"], sub["pred_eform"])
        mse  = mean_squared_error(sub["true_eform"], sub["pred_eform"])
        rmse = np.sqrt(mse)
        color = elem_colors.get(elem, "#555555")

        ax = fig.add_subplot(gs[idx])
        ax.scatter(
            sub["true_eform"],
            sub["pred_eform"],
            alpha=0.75,
            color=color,
            edgecolors="k",
            linewidths=0.3,
            s=40,
            zorder=3,
            label=f"{elem} ({len(sub)} pts)",
        )

        min_val = min(sub["true_eform"].min(), sub["pred_eform"].min())
        max_val = max(sub["true_eform"].max(), sub["pred_eform"].max())
        margin  = 0.05 * (max_val - min_val)
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                "r--", lw=1.5, zorder=2)

        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        ax.set_xlabel("True $E_f$ (eV/atom)", fontsize=11)
        ax.set_ylabel("Predicted $E_f$ (eV/atom)", fontsize=11)
        ax.set_title(
            f"Element: {elem}\n"
            f"MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}",
            fontsize=10,
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "MEGNet Fine-Tuned: Parity Plots — B, C, N Test Sets\n"
        "(Hybrid 80/20 split per element, combined training)",
        fontsize=13, y=1.02,
    )
    combined_plot = os.path.join(OUTPUT_DIR, "parity_plot_BCN_hybrid.png")
    fig.savefig(combined_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCombined parity plot saved: {combined_plot}")

# ============================================================
# STEP 6: Summary metrics table
# ============================================================
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUT_DIR, "summary_metrics_BCN_hybrid.csv")
summary_df.to_csv(summary_csv, index=False)

print("\n" + "=" * 60)
print(" SUMMARY — MEGNet Hybrid B+C+N Fine-Tune Results")
print("=" * 60)
print(summary_df.to_string(index=False))
print(f"\nSummary saved to: {summary_csv}")
print("\nAll outputs written to:", os.path.abspath(OUTPUT_DIR))