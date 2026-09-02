import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from megnet.utils.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# ===============================
# USER CONFIG
# ===============================
CIF_DIR       = "test_cif/C_materials"            # folder containing Carbon CIF files
CSV_PATH      = os.path.join(CIF_DIR, "id_prop_Carbon.csv")  # CSV with 'id' & 'formation_energy'
OUTPUT_CSV    = "megnet_results_C_finetune.csv"
SEED          = 42
EPOCHS        = 50           # fine-tuning epochs
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4

# ===============================
# LOAD PRETRAINED MEGNet MODEL
# ===============================
print("Loading pretrained MEGNet model...")
model = load_model("Eform_MP_2018")
print("Model loaded.\n")

# ===============================
# LOAD CSV
# ===============================
df = pd.read_csv(CSV_PATH)
required_cols = {"id", "formation_energy"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")
print(f"Total samples in CSV: {len(df)}")

# ===============================
# FIXED 80/20 SPLIT
# ===============================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
print(f"Train samples : {len(train_df)}")
print(f"Test  samples : {len(test_df)}\n")

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

# ===============================
# LOAD TRAIN & TEST STRUCTURES
# ===============================
print("Loading training structures...")
train_structures, train_targets, train_ids = load_structures_and_targets(
    train_df, CIF_DIR, label="TRAIN"
)
print("\nLoading test structures...")
test_structures, test_targets, test_ids = load_structures_and_targets(
    test_df, CIF_DIR, label="TEST"
)
if len(train_structures) == 0:
    raise RuntimeError("No training structures could be loaded. Check CIF_DIR and CSV.")

# ===============================
# RECOMPILE MODEL WITH LOWER LR
# ===============================
print("\nRecompiling model for fine-tuning...")
model.model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mae"
)
print(f"  Optimizer : Adam  |  LR = {LEARNING_RATE}  |  Loss = MAE")

# ===============================
# FINE-TUNE ON TRAINING SET
# ===============================
print(f"\nFine-tuning for {EPOCHS} epochs (batch_size={BATCH_SIZE})...\n")
history = model.train(
    train_structures,
    train_targets,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    scrub_failed_structures=True,
)
print("\nFine-tuning complete.\n")

# ===============================
# PREDICTION ON TEST SET
# ===============================
results = []
print("Running predictions on test set...\n")
for structure, true_eform, material_id in zip(test_structures, test_targets, test_ids):
    try:
        pred_eform = float(model.predict_structure(structure).ravel()[0])
        results.append({
            "id"        : material_id,
            "true_eform": true_eform,
            "pred_eform": pred_eform,
            "error"     : pred_eform - true_eform,
        })
        print(f"  {material_id} | True = {true_eform:.4f} | Pred = {pred_eform:.4f} "
              f"| Err = {pred_eform - true_eform:+.4f}")
    except Exception as e:
        print(f"  [ERROR] {material_id} -> {e}")

# ===============================
# SAVE RESULTS
# ===============================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to: {OUTPUT_CSV}")

# ===============================
# EVALUATION METRICS
# ===============================
mae  = mean_absolute_error(results_df["true_eform"], results_df["pred_eform"])
rmse = np.sqrt(mean_squared_error(results_df["true_eform"], results_df["pred_eform"]))
print("\n=================================")
print(" MEGNet Carbon Fine-Tune Results ")
print("=================================")
print(f"  Test samples : {len(results_df)}")
print(f"  MAE          : {mae:.4f}  eV/atom")
print(f"  RMSE         : {rmse:.4f} eV/atom")

# ===============================
# TRAINING LOSS CURVE
# ===============================
if history is not None:
    try:
        loss_key = [k for k in history.history.keys() if "loss" in k.lower()][0]
        plt.figure(figsize=(7, 4))
        plt.plot(history.history[loss_key], label="Train Loss (MAE)")
        if "val_" + loss_key in history.history:
            plt.plot(history.history["val_" + loss_key], label="Val Loss (MAE)")
        plt.xlabel("Epoch")
        plt.ylabel("MAE Loss")
        plt.title("MEGNet Fine-Tuning Loss Curve — Carbon")
        plt.legend()
        plt.tight_layout()
        plt.savefig("finetune_loss_curve_MEGNet_C.png", dpi=300)
        plt.show()
        print("Loss curve saved: finetune_loss_curve_MEGNet_C.png")
    except Exception as e:
        print(f"[INFO] Could not plot loss curve: {e}")

# ===============================
# PARITY PLOT
# ===============================
plt.figure(figsize=(6, 6))
plt.scatter(results_df["true_eform"], results_df["pred_eform"],
            alpha=0.7, edgecolors="k", linewidths=0.4)
min_val = min(results_df["true_eform"].min(), results_df["pred_eform"].min())
max_val = max(results_df["true_eform"].max(), results_df["pred_eform"].max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Ideal")
plt.xlabel("True Formation Energy (eV/atom)")
plt.ylabel("Predicted Formation Energy (eV/atom)")
plt.title(f"MEGNet Fine-Tuned — Carbon Test Set\nMAE = {mae:.4f} eV/atom | RMSE = {rmse:.4f} eV/atom")
plt.legend()
plt.tight_layout()
plt.savefig("Formation_energy_comparison_MEGNet_C_finetune.png", dpi=300)
plt.show()
print("Parity plot saved: Formation_energy_comparison_MEGNet_C_finetune.png")