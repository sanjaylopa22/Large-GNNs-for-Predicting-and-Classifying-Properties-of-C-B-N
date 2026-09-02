#!/usr/bin/env python
# Domain-specific fine-tuning of ALIGNN for Nitrogen
# 80% training / 20% testing (random split, reproducible)
# Saves CSVs and plots for evaluation

#!/usr/bin/env python
# Domain-specific fine-tuning of ALIGNN for Nitrogen
# 80% training / 20% testing (random split, reproducible)
# Saves CSV + plots + MAE/MSE

import torch
import os
import json
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_NAME = "jv_formation_energy_peratom_alignn"

TRAIN_CIF_DIR = "./examples/sample_data/N_materials"
ID_PROP_CSV = os.path.join(TRAIN_CIF_DIR, "id_prop_Nitrogen.csv")

OUTPUT_MODEL = "alignn_finetuned_nitrogen.pth"

PREDICTIONS_CSV = "nitrogen_test_predictions.csv"
MERGED_CSV = "nitrogen_predictions_vs_gt.csv"
SCATTER_PLOT = "scatter_plot_nitrogen.png"
COMPARISON_PLOT = "comparison_plot_nitrogen.png"

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
CUTOFF = 8.0
MAX_NEIGHBORS = 12
FREEZE_BACKBONE = True
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# FIGSHARE MODEL INFO
# --------------------------------------------------

ALL_MODELS = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679",
        1,
    ]
}


# --------------------------------------------------
# LOAD PRETRAINED ALIGNN
# --------------------------------------------------

def get_figshare_model(model_name):

    url, _ = ALL_MODELS[model_name]
    zip_path = model_name + ".zip"

    if not os.path.exists(zip_path):
        print("Downloading pretrained ALIGNN...")
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as z:

        chk = [x for x in z.namelist() if "checkpoint_" in x][0]
        cfg = [x for x in z.namelist() if "config.json" in x][0]

        config = json.loads(z.read(cfg))
        state = z.read(chk)

        model = ALIGNN(ALIGNNConfig(**config["model"]))

        fd, tmp = tempfile.mkstemp()

        with open(tmp, "wb") as f:
            f.write(state)

        ckpt = torch.load(tmp, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model"])

        os.remove(tmp)

    model.to(device)

    return model


# --------------------------------------------------
# ALIGNN DATASET
# --------------------------------------------------

class ALIGNNDataset(Dataset):

    def __init__(self, cif_dir, id_prop_csv):

        self.cif_dir = cif_dir

        df = pd.read_csv(id_prop_csv)

        if "file_name" not in df.columns or "formation_energy_per_atom" not in df.columns:
            raise ValueError("CSV must contain file_name and formation_energy_per_atom")

        self.data = df[["file_name", "formation_energy_per_atom"]].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        cif_name, target = self.data[idx]

        if not cif_name.endswith(".cif"):
            cif_name += ".cif"

        cif_path = os.path.join(self.cif_dir, cif_name)

        atoms = Atoms.from_cif(cif_path)

        g, lg = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=CUTOFF,
            max_neighbors=MAX_NEIGHBORS,
        )

        lattice = torch.tensor(
            atoms.lattice_mat,
            dtype=torch.float32
        )

        y = torch.tensor(
            float(target),
            dtype=torch.float32
        )

        return g, lg, lattice, y, cif_name


# --------------------------------------------------
# COLLATE
# --------------------------------------------------

def collate_alignn(batch):

    import dgl

    gs, lgs, lats, ys, names = zip(*batch)

    return (
        dgl.batch(gs),
        dgl.batch(lgs),
        torch.stack(lats),
        torch.stack(ys),
        names,
    )


# --------------------------------------------------
# FINETUNE
# --------------------------------------------------

def finetune_nitrogen():

    print("\n" + "=" * 60)
    print(" Fine-tuning ALIGNN on Nitrogen ")
    print("=" * 60)

    dataset = ALIGNNDataset(
        TRAIN_CIF_DIR,
        ID_PROP_CSV,
    )

    n_total = len(dataset)

    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    train_set, test_set = random_split(
        dataset,
        [n_train, n_test],
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_alignn,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_alignn,
    )

    model = get_figshare_model(MODEL_NAME)

    # Freeze backbone
    if FREEZE_BACKBONE:

        for name, param in model.named_parameters():

            if "fc" not in name:
                param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    criterion = torch.nn.MSELoss()

    # ---------------- TRAIN ----------------

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0

        for g, lg, lat, y, _ in train_loader:

            g = g.to(device)
            lg = lg.to(device)
            lat = lat.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model([g, lg, lat])

            loss = criterion(
                pred.view(-1),
                y.view(-1),
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss={total_loss/len(train_loader):.6f}"
        )

    torch.save(
        model.state_dict(),
        OUTPUT_MODEL,
    )

    print("Saved:", OUTPUT_MODEL)

    # ---------------- TEST ----------------

    model.eval()

    y_true_list = []
    y_pred_list = []
    names_list = []

    with torch.no_grad():

        for g, lg, lat, y, names in test_loader:

            g = g.to(device)
            lg = lg.to(device)
            lat = lat.to(device)
            y = y.to(device)

            pred = model([g, lg, lat])

            y_true_list.extend(
                y.cpu().numpy()
            )

            y_pred_list.extend(
                pred.cpu().numpy()
            )

            names_list.extend(names)

    # ---------------- CSV ----------------

    df = pd.DataFrame(
        {
            "file_name": names_list,
            "predicted_energy_per_atom": y_pred_list,
            "formation_energy_per_atom": y_true_list,
        }
    )

    df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    df.to_csv(
        MERGED_CSV,
        index=False,
    )

    print("Saved CSV")

    # ---------------- METRICS ----------------

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    mse = np.mean(
        (y_true - y_pred) ** 2
    )

    print("\n========== NITROGEN ==========")
    print("Total:", n_total)
    print("Train:", n_train)
    print("Test:", n_test)
    print("MAE:", mae)
    print("MSE:", mse)

    # ---------------- SCATTER ----------------

    plt.figure()

    plt.scatter(
        y_true,
        y_pred,
    )

    plt.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
    )

    plt.savefig(
        SCATTER_PLOT
    )

    # ---------------- BAR ----------------

    plt.figure()

    x = np.arange(len(y_true))

    plt.bar(
        x,
        y_true,
        alpha=0.5,
    )

    plt.bar(
        x,
        y_pred,
        alpha=0.5,
    )

    plt.savefig(
        COMPARISON_PLOT
    )

    # ---------------- PRINT ----------------

    print("\nPredictions")

    for i in range(len(y_true)):

        print(
            names_list[i],
            y_pred[i],
            y_true[i],
        )


# --------------------------------------------------

if __name__ == "__main__":
    finetune_nitrogen()




'''
import torch
import os
import json
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
MODEL_NAME = "jv_formation_energy_peratom_alignn"

TRAIN_CIF_DIR = "./examples/sample_data/N_materials"
ID_PROP_CSV = os.path.join(TRAIN_CIF_DIR, "id_prop_Nitrogen.csv")

OUTPUT_MODEL = "alignn_finetuned_nitrogen.pth"
PREDICTIONS_CSV = "nitrogen_test_predictions.csv"
MERGED_CSV = "nitrogen_predictions_vs_gt.csv"
SCATTER_PLOT = "scatter_plot_N_finetune.png"
COMPARISON_PLOT = "comparison_plot_N_finetune.png"

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
CUTOFF = 8.0
MAX_NEIGHBORS = 12
FREEZE_BACKBONE = True
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# FIGSHARE MODEL INFO
# --------------------------------------------------
ALL_MODELS = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679",
        1,
    ]
}

# --------------------------------------------------
# LOAD PRETRAINED ALIGNN
# --------------------------------------------------
def get_figshare_model(model_name):
    url, _ = ALL_MODELS[model_name]
    zip_path = model_name + ".zip"

    if not os.path.exists(zip_path):
        print("Downloading pretrained ALIGNN...")
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as z:
        chk = [x for x in z.namelist() if "checkpoint_" in x][0]
        cfg = [x for x in z.namelist() if "config.json" in x][0]

        config = json.loads(z.read(cfg))
        state = z.read(chk)

        model = ALIGNN(ALIGNNConfig(**config["model"]))

        fd, tmp = tempfile.mkstemp()
        with open(tmp, "wb") as f:
            f.write(state)

        ckpt = torch.load(tmp, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        os.remove(tmp)

    model.to(device)
    return model

# --------------------------------------------------
# ALIGNN DATASET
# --------------------------------------------------
class ALIGNNDataset(Dataset):
    def __init__(self, cif_dir, id_prop_csv):
        self.cif_dir = cif_dir
        df = pd.read_csv(id_prop_csv)
        if "file_name" not in df.columns or "formation_energy_per_atom" not in df.columns:
            raise ValueError("CSV must contain 'file_name' and 'formation_energy_per_atom' columns")
        self.data = df[["file_name", "formation_energy_per_atom"]].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cif_name, target = self.data[idx]
        if not cif_name.lower().endswith(".cif"):
            cif_name += ".cif"
        cif_path = os.path.join(self.cif_dir, cif_name)
        if not os.path.exists(cif_path):
            raise FileNotFoundError(f"{cif_path} not found")

        atoms = Atoms.from_cif(cif_path)
        g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=CUTOFF, max_neighbors=MAX_NEIGHBORS)
        lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
        y = torch.tensor(float(target), dtype=torch.float32)
        return g, lg, lattice, y, cif_name

# --------------------------------------------------
# COLLATE FUNCTION
# --------------------------------------------------
def collate_alignn(batch):
    import dgl
    gs, lgs, lats, ys, names = zip(*batch)
    return dgl.batch(gs), dgl.batch(lgs), torch.stack(lats), torch.stack(ys), names

# --------------------------------------------------
# FINETUNING & EVALUATION
# --------------------------------------------------
def finetune_nitrogen():

    print("\n" + "=" * 60)
    print(" Fine-tuning ALIGNN on Nitrogen Formation Energy ")
    print("=" * 60)

    dataset = ALIGNNDataset(TRAIN_CIF_DIR, ID_PROP_CSV)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_alignn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_alignn)

    model = get_figshare_model(MODEL_NAME)

    # Freeze backbone
    if FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # ---------------- Training ----------------
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for g, lg, lat, y, _ in train_loader:
            g, lg, lat, y = g.to(device), lg.to(device), lat.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model([g, lg, lat])
            loss = criterion(pred.view(-1), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1:03d}/{EPOCHS}]  Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), OUTPUT_MODEL)
    print("\n✔ Fine-tuned Nitrogen model saved to:", OUTPUT_MODEL)

    # ---------------- Evaluation ----------------
    print("\nEvaluating on test set...")
    model.eval()
    y_true_list, y_pred_list, names_list = [], [], []

    with torch.no_grad():
        for g, lg, lat, y, names in test_loader:
            g, lg, lat, y = g.to(device), lg.to(device), lat.to(device), y.to(device)
            pred = model([g, lg, lat])
            y_true_list.extend(y.view(-1).cpu().numpy())
            y_pred_list.extend(pred.view(-1).cpu().numpy())
            names_list.extend(names)

    # ---- Save predictions CSV ----
    pred_df = pd.DataFrame({
        "file_name": names_list,
        "predicted_energy_per_atom": y_pred_list,
        "formation_energy_per_atom": y_true_list
    })
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nPredictions saved to {PREDICTIONS_CSV}")

    # ---- Save merged CSV ----
    merged_df = pred_df.copy()
    merged_df.to_csv(MERGED_CSV, index=False)
    print(f"Merged predictions vs GT saved to {MERGED_CSV}")

    # ---- Scatter plot ----
    plt.figure(figsize=(6,6))
    plt.scatter(y_true_list, y_pred_list, color='green', alpha=0.7)
    plt.plot([min(y_true_list), max(y_true_list)], [min(y_true_list), max(y_true_list)], 'r--', label='y=x')
    plt.xlabel("Ground Truth Formation Energy (eV/atom)")
    plt.ylabel("Predicted Formation Energy (eV/atom)")
    plt.title("ALIGNN Prediction vs Ground Truth (Nitrogen)")
    plt.legend()
    plt.savefig(SCATTER_PLOT, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {SCATTER_PLOT}")

    # ---- Comparison bar plot ----
    plt.figure(figsize=(10,6))
    x = np.arange(len(y_true_list))
    width = 0.35
    plt.bar(x - width/2, y_true_list, width, label='Ground Truth')
    plt.bar(x + width/2, y_pred_list, width, label='Predicted')
    plt.xlabel("Material Index")
    plt.ylabel("Formation Energy (eV/atom)")
    plt.title("Ground Truth vs Predicted Formation Energy (Nitrogen)")
    plt.legend()
    plt.savefig(COMPARISON_PLOT, dpi=300)
    plt.close()
    print(f"Comparison plot saved to {COMPARISON_PLOT}")

    # ---- Print individual results ----
    print("\n==================== INDIVIDUAL PREDICTIONS ====================")
    for i in range(len(y_true_list)):
        print(f"{names_list[i]}: Pred = {y_pred_list[i]:.4f}, GT = {y_true_list[i]:.4f}")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    finetune_nitrogen()
'''