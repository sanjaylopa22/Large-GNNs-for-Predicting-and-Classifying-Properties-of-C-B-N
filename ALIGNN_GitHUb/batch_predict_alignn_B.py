#!/usr/bin/env python
import os
import json
import torch
import zipfile
import tempfile
import requests
from tqdm import tqdm
import pandas as pd
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#torch.serialization.add_safe_globals([getattr])

# ---------------- SETTINGS ---------------- #
MODEL_NAME = "jv_formation_energy_peratom_alignn"
DATA_FOLDER = "./examples/sample_data/B_materials"
GROUND_TRUTH_CSV = "./examples/sample_data/B_materials/id_prop_Boron.csv"

OUTPUT_CSV = "alignn_predictions_Boron_test.csv"
MERGED_CSV = "alignn_predictions_vs_gt_Boron.csv"
FAILED_LOG = "failed_cifs_test.log"

TEST_SIZE = 0.2
RANDOM_SEED = 42

# ---------------- DEVICE ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL DICT ---------------- #
all_models = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679",
        1,
    ]
}

# ---------------- UTILITIES ---------------- #
def clean_name(x):
    return str(x).strip().replace(".cif", "")

def get_figshare_model(model_name="jv_formation_energy_peratom_alignn"):
    url, _ = all_models[model_name]
    zfile = model_name + ".zip"
    path = os.path.join(os.path.dirname(__file__), zfile)

    if not os.path.isfile(path):
        print(f"Downloading {model_name} from Figshare...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(path, "wb") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()

    with zipfile.ZipFile(path) as zp:
        chk_file = [i for i in zp.namelist() if "checkpoint_" in i and "pt" in i][0]
        cfg_file = [i for i in zp.namelist() if "config.json" in i][0]

        config = json.loads(zp.read(cfg_file))
        data = zp.read(chk_file)
        model = ALIGNN(ALIGNNConfig(**config["model"]))

        tmp_file, tmp_path = tempfile.mkstemp()
        with open(tmp_path, "wb") as f:
            f.write(data)

        model.load_state_dict(
            torch.load(tmp_path, map_location=device, weights_only=False)["model"]
        )
        model.to(device)
        model.eval()
        os.remove(tmp_path)

    return model

def get_prediction(atoms, cutoff=8, max_neighbors=12, model=None):
    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(cutoff), max_neighbors=max_neighbors)
    lat = torch.tensor(atoms.lattice_mat)
    out_data = model([g.to(device), lg.to(device), lat.to(device)]).detach().cpu().numpy().flatten()
    return out_data[0]

# ---------------- MAIN SCRIPT ---------------- #
if __name__ == "__main__":
    # ---- Load CIF files ---- #
    cif_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".cif")]
    cif_files.sort()
    print(f"Found {len(cif_files)} CIF files")

    # ---- Load ground truth ---- #
    if os.path.exists(GROUND_TRUTH_CSV):
        gt_df = pd.read_csv(GROUND_TRUTH_CSV)
        gt_df["file_name"] = gt_df["file_name"].apply(clean_name)
        valid_files = set(gt_df["file_name"])
        cif_files = [f for f in cif_files if clean_name(os.path.basename(f)) in valid_files]
        print(f"{len(cif_files)} CIF files match ground-truth")
    else:
        gt_df = pd.DataFrame(columns=["file_name", "formation_energy_per_atom"])
        print("No ground-truth CSV found. Predictions only.")

    # ---- Split test set (20%) ---- #
    _, test_files = train_test_split(cif_files, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    print(f"Using {len(test_files)} files for testing")

    # ---- Load model ---- #
    model = get_figshare_model(MODEL_NAME)

    results = []
    failed_files = []

    # ---- Prediction loop ---- #
    for cif_path in test_files:
        file_name = clean_name(os.path.basename(cif_path))
        print(f"Predicting: {file_name}")
        try:
            atoms = Atoms.from_cif(cif_path)
            pred = get_prediction(atoms, model=model)
            results.append([file_name, pred])
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            results.append([file_name, None])
            failed_files.append(f"{file_name}: {e}")

    # ---- Save predictions ---- #
    pred_df = pd.DataFrame(results, columns=["file_name", "predicted_energy_per_atom"])
    pred_df.to_csv(OUTPUT_CSV, index=False)
    with open(FAILED_LOG, "w") as f:
        f.write("\n".join(failed_files))
    print(f"\nPredictions saved to {OUTPUT_CSV}")
    print(f"Failed CIFs logged to {FAILED_LOG}")

    # ---- Merge with ground truth ---- #
    if not gt_df.empty:
        merged = pred_df.merge(gt_df, on="file_name", how="inner")
        merged["predicted_energy_per_atom"] = pd.to_numeric(merged["predicted_energy_per_atom"], errors="coerce")
        merged["formation_energy_per_atom"] = pd.to_numeric(merged["formation_energy_per_atom"], errors="coerce")

        valid_mask = merged["predicted_energy_per_atom"].notnull()
        y_true = merged.loc[valid_mask, "formation_energy_per_atom"].values
        y_pred = merged.loc[valid_mask, "predicted_energy_per_atom"].values

        # ---- Compute MAE and MSE ---- #
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        print("\n==================== ERROR METRICS ====================")
        print(f"MAE: {mae:.4f} eV/atom")
        print(f"MSE: {mse:.6f} (eV/atom)^2")
        print(f"Valid structures: {len(y_true)}")

        # ---- Save merged CSV ---- #
        merged.to_csv(MERGED_CSV, index=False)

        # ---- Scatter plot ---- #
        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, color='blue', alpha=0.7)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='y=x')
        plt.xlabel("Ground Truth Formation Energy (eV/atom)")
        plt.ylabel("Predicted Formation Energy (eV/atom)")
        plt.title("ALIGNN Prediction vs Ground Truth")
        plt.legend()
        plt.savefig("scatter_plot_B_pretrained.png", dpi=300)
        plt.close()

        # ---- Comparison bar plot ---- #
        plt.figure(figsize=(10,6))
        x = np.arange(len(y_true))
        width = 0.35
        plt.bar(x - width/2, y_true, width, label='Ground Truth')
        plt.bar(x + width/2, y_pred, width, label='Predicted')
        plt.xlabel("Material Index")
        plt.ylabel("Formation Energy (eV/atom)")
        plt.title("Ground Truth vs Predicted Formation Energy")
        plt.legend()
        plt.savefig("comparison_plot_B_pretrained.png", dpi=300)
        plt.close()

        # ---- Print predictions ---- #
        print("\n==================== PREDICTIONS ====================")
        for _, row in merged.iterrows():
            print(f"{row['file_name']}: Pred = {row['predicted_energy_per_atom']:.4f}, GT = {row['formation_energy_per_atom']:.4f}")

    else:
        print("\n==================== PREDICTIONS ====================")
        for _, row in pred_df.iterrows():
            print(f"{row['file_name']}: Pred = {row['predicted_energy_per_atom']:.4f}")