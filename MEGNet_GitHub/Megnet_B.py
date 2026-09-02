import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from megnet.utils.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================
# USER CONFIG
# ===============================
CIF_DIR    = "test_cif/B_materials"                    # folder containing Boron CIF files
CSV_PATH   = os.path.join(CIF_DIR, "id_prop_Boron.csv") # CSV with 'id' & 'formation_energy'
OUTPUT_CSV = "megnet_results_B_test.csv"
SEED       = 42

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
# PREDICTION LOOP ON TEST SET
# ===============================
results = []
print("Running predictions on test set...\n")

for idx, row in test_df.iterrows():
    material_id = str(row["id"])
    true_eform  = float(row["formation_energy"])
    cif_file    = material_id if material_id.endswith(".cif") else f"{material_id}.cif"
    cif_path    = os.path.join(CIF_DIR, cif_file)

    if not os.path.exists(cif_path):
        print(f"  [WARNING] Missing CIF: {cif_path}")
        continue

    try:
        structure  = Structure.from_file(cif_path)
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
mse  = mean_squared_error(results_df["true_eform"],  results_df["pred_eform"])
rmse = np.sqrt(mse)

print("\n==========================")
print(" MEGNet Boron Test Results")
print("==========================")
print(f"  Samples : {len(results_df)}")
print(f"  MAE     : {mae:.4f}  eV/atom")
print(f"  RMSE    : {rmse:.4f} eV/atom")
print(f"  MSE     : {mse:.4f}  (eV/atom)^2")

# ===============================
# PARITY PLOT
# ===============================
plt.figure(figsize=(6, 6))
plt.scatter(
    results_df["true_eform"],
    results_df["pred_eform"],
    alpha=0.7, edgecolors="k", linewidths=0.4
)

min_val = min(results_df["true_eform"].min(), results_df["pred_eform"].min())
max_val = max(results_df["true_eform"].max(), results_df["pred_eform"].max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Ideal (y = x)")

plt.xlabel("True Formation Energy (eV/atom)")
plt.ylabel("Predicted Formation Energy (eV/atom)")
plt.title(
    f"MEGNet Pretrained — Boron Test Set\n"
    f"MAE = {mae:.4f} eV/atom  |  RMSE = {rmse:.4f} eV/atom"
)
plt.legend()
plt.tight_layout()
plt.savefig("Formation_energy_comparison_MEGNet_B.png", dpi=300)
plt.show()
print("Parity plot saved: Formation_energy_comparison_MEGNet_B.png")

'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from megnet.utils.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================
# USER CONFIG
# ===============================
CIF_DIR = "test_cif/B_materials"           # folder containing Boron CIF files
CSV_PATH = os.path.join(CIF_DIR, "id_prop_Boron.csv")  # CSV with 'id' & 'formation_energy'
OUTPUT_CSV = "megnet_results_B_test.csv"
SEED = 42

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

# ===============================
# SPLIT 80/20
# ===============================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
print(f"Total samples: {len(df)}, Test samples: {len(test_df)}")

# ===============================
# PREDICTION LOOP ON TEST SET
# ===============================
results = []
print("\nRunning predictions on test set...\n")

for idx, row in test_df.iterrows():
    material_id = str(row["id"])
    true_eform = float(row["formation_energy"])

    cif_file = material_id if material_id.endswith(".cif") else f"{material_id}.cif"
    cif_path = os.path.join(CIF_DIR, cif_file)

    if not os.path.exists(cif_path):
        print(f"[WARNING] Missing CIF: {cif_path}")
        continue

    try:
        structure = Structure.from_file(cif_path)
        pred_eform = model.predict_structure(structure).ravel()[0]

        results.append({
            "id": material_id,
            "true_eform": true_eform,
            "pred_eform": pred_eform,
            "error": pred_eform - true_eform
        })

        print(f"{material_id} | True = {true_eform:.4f} | Pred = {pred_eform:.4f}")

    except Exception as e:
        print(f"[ERROR] {material_id} -> {e}")

# ===============================
# SAVE RESULTS
# ===============================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to: {OUTPUT_CSV}")

# ===============================
# EVALUATION METRICS
# ===============================
mae = mean_absolute_error(results_df["true_eform"], results_df["pred_eform"])
mse = mean_squared_error(results_df["true_eform"], results_df["pred_eform"])

print("\n==========================")
print("MEGNet Boron Test Results")
print("==========================")
print(f"Samples: {len(results_df)}")
print(f"MAE: {mae:.4f} eV/atom")
print(f"MSE: {mse:.4f} (eV/atom)^2")

# ===============================
# PARITY PLOT
# ===============================
plt.figure(figsize=(6,6))
plt.scatter(results_df["true_eform"], results_df["pred_eform"], alpha=0.7)

min_val = min(results_df["true_eform"].min(), results_df["pred_eform"].min())
max_val = max(results_df["true_eform"].max(), results_df["pred_eform"].max())
plt.plot([min_val, max_val], [min_val, max_val], "k--")

plt.xlabel("True Formation Energy (eV/atom)")
plt.ylabel("Predicted Formation Energy (eV/atom)")
plt.title("MEGNet Boron Test Prediction")
plt.tight_layout()
plt.savefig("Formation_energy_comparison_MEGNet_B.png", dpi=300)
plt.show()
print("Scatter plot saved: Formation_energy_comparison_MEGNet_B.png")
'''
