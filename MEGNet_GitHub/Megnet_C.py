import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from megnet.utils.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================
# USER CONFIG
# ===============================
CIF_DIR = "test_cif/C_materials"
CSV_PATH = os.path.join(CIF_DIR, "id_prop_Carbon.csv")
OUTPUT_CSV = "megnet_results_C_test.csv"
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
print("MEGNet Test Results")
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
plt.title("MEGNet Carbon Test Prediction")
plt.tight_layout()
plt.savefig("Formation_energy_comparison_MEGNet_C.png", dpi=300)
plt.show()
print("Scatter plot saved: Formation_energy_comparison_MEGNet_C.png")

'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from megnet.data.crystal import CrystalGraph
from megnet.data.crystal import get_elemental_embeddings
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from megnet.layers import MEGNetLayer, Set2Set

# ======================================
# USER CONFIG
# ======================================
CIF_DIR = "test_cif/C_materials"      # Folder containing CIF files
CSV_PATH = os.path.join(CIF_DIR, "id_prop_Carbon.csv")  # CSV with 'id' & 'formation_energy'
OUTPUT_CSV = "custom_megnet_results_C.csv"

# Hyperparameters
n_atom_feature = 20
n_bond_feature = 10
n_global_feature = 2

# ======================================
# HELPER: Load CIFs
# ======================================
def load_cif_structures(cif_dir, csv_path):
    df = pd.read_csv(csv_path)
    structures = []
    targets = []
    ids = []

    for idx, row in df.iterrows():
        material_id = row["id"]
        target = row["formation_energy"]
        cif_file = material_id if material_id.endswith(".cif") else f"{material_id}.cif"
        cif_path = os.path.join(cif_dir, cif_file)
        if not os.path.exists(cif_path):
            print(f"[WARNING] CIF not found: {cif_path}")
            continue
        try:
            struct = Structure.from_file(cif_path)
            structures.append(struct)
            targets.append(target)
            ids.append(material_id)
        except Exception as e:
            print(f"[ERROR] {material_id}: {e}")
    return structures, np.array(targets), ids

# ======================================
# CREATE CRYSTAL GRAPHS
# ======================================
print("Loading structures...")
structures, targets, ids = load_cif_structures(CIF_DIR, CSV_PATH)
print(f"Loaded {len(structures)} structures.")

# Load elemental embeddings (optional transfer learning)
el_embeddings = get_elemental_embeddings()  # dict: element -> vector

# Create CrystalGraph objects
graphs = [CrystalGraph(struct, el_embeddings=el_embeddings) for struct in structures]

# MEGNet requires inputs: x (atom features), a (bond features), u (global features), 
#   and indices for graph edges (i, j, bond_indices, atom_indices)
# We'll prepare lists for these
x_list, a_list, u_list, i_list, j_list, atom_list, bond_list = [], [], [], [], [], [], []

for g in graphs:
    x_list.append(np.array(g.node_features, dtype=np.float32))
    a_list.append(np.array(g.edge_features, dtype=np.float32))
    u_list.append(np.array(g.global_features, dtype=np.float32))
    i_list.append(np.array(g.edge_index[0], dtype=np.int32))
    j_list.append(np.array(g.edge_index[1], dtype=np.int32))
    atom_list.append(np.array(g.node_index, dtype=np.int32))
    bond_list.append(np.array(g.edge_index_map, dtype=np.int32))

# ======================================
# BUILD CUSTOM MEGNet MODEL
# ======================================
# Inputs
x1 = Input(shape=(None, n_atom_feature))
x2 = Input(shape=(None, n_bond_feature))
x3 = Input(shape=(None, n_global_feature))
x4 = Input(shape=(None,), dtype='int32')
x5 = Input(shape=(None,), dtype='int32')
x6 = Input(shape=(None,), dtype='int32')
x7 = Input(shape=(None,), dtype='int32')
inputs = [x1, x2, x3, x4, x5, x6, x7]

# MEGNetLayer
out_v, out_e, out_u = MEGNetLayer(
    units_v=[32, 16],
    units_e=[32, 16],
    units_u=[32, 16],
    pool_method='mean',
    activation='relu'
)(inputs)

# Output per structure
output = Dense(1)(out_u)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mse')
print(model.summary())

# ======================================
# PREDICTION LOOP
# ======================================
preds = []
for xi, ai, ui, ii, ji, atomi, bondi in zip(x_list, a_list, u_list, i_list, j_list, atom_list, bond_list):
    # Expand dims to batch size 1
    xi = np.expand_dims(xi, 0)
    ai = np.expand_dims(ai, 0)
    ui = np.expand_dims(ui, 0)
    ii = np.expand_dims(ii, 0)
    ji = np.expand_dims(ji, 0)
    atomi = np.expand_dims(atomi, 0)
    bondi = np.expand_dims(bondi, 0)

    pred = model.predict([xi, ai, ui, ii, ji, atomi, bondi], verbose=0)
    preds.append(pred[0, 0])

preds = np.array(preds)

# ======================================
# METRICS
# ======================================
mae = mean_absolute_error(targets, preds)
mse = mean_squared_error(targets, preds)
print(f"\nNumber of materials: {len(preds)}")
print(f"MAE: {mae:.4f} eV/atom")
print(f"MSE: {mse:.4f} (eV/atom)^2")

# Save CSV
results_df = pd.DataFrame({"id": ids, "true_eform": targets, "pred_eform": preds})
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to: {OUTPUT_CSV}")

# Scatter plot
plt.figure(figsize=(6,6))
plt.scatter(targets, preds, alpha=0.7)
plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--')
plt.xlabel("True Formation Energy (eV/atom)")
plt.ylabel("Predicted Formation Energy (eV/atom)")
plt.title("Custom MEGNet Formation Energy Prediction")
plt.tight_layout()
plt.savefig("custom_megnet_scatter.png", dpi=300)
plt.close()
print("Scatter plot saved: custom_megnet_scatter.png")
'''