#!/usr/bin/env python
"""
Ablation (Point 2): ALIGNN fine-tuning with UNFROZEN backbone
================================================================
Identical to the corrected finetuned_alignn_{B,C,N}.py scripts
(MAE loss, 80/20 split seed=42), except FREEZE_BACKBONE = False,
so the full network is fine-tuned rather than only the output head.
This isolates how much of ALIGNN's reported performance gap versus
CGCNN/MEGNet is attributable to the frozen-backbone protocol rather
than the architecture itself.

Loops over all three elements in one run. Outputs go to
output_ALIGNN_unfrozen_ablation/<element>/.
"""

import torch
import os
import json
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig

# ===============================
# USER CONFIG
# ===============================
MODEL_NAME = "jv_formation_energy_peratom_alignn"

ELEMENTS = ["B", "C", "N"]
CIF_DIRS = {
    "B": "./examples/sample_data/B_materials",
    "C": "./examples/sample_data/C_materials",
    "N": "./examples/sample_data/N_materials",
}
CSV_NAMES = {
    "B": "id_prop_Boron.csv",
    "C": "id_prop_Carbon.csv",
    "N": "id_prop_Nitrogen.csv",
}

OUTPUT_ROOT = "output_ALIGNN_unfrozen_ablation"

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
CUTOFF = 8.0
MAX_NEIGHBORS = 12
FREEZE_BACKBONE = False   # <-- the ablation: full-network fine-tuning
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_MODELS = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679", 1,
    ]
}


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


class ALIGNNDataset(Dataset):
    def __init__(self, cif_dir, id_prop_csv):
        self.cif_dir = cif_dir
        df = pd.read_csv(id_prop_csv)
        self.data = df[["file_name", "formation_energy_per_atom"]].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cif_name, target = self.data[idx]
        if not cif_name.endswith(".cif"):
            cif_name += ".cif"
        path = os.path.join(self.cif_dir, cif_name)
        atoms = Atoms.from_cif(path)
        g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=CUTOFF, max_neighbors=MAX_NEIGHBORS)
        lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
        y = torch.tensor(float(target), dtype=torch.float32)
        return g, lg, lattice, y, cif_name


def collate_alignn(batch):
    import dgl
    gs, lgs, lats, ys, names = zip(*batch)
    return dgl.batch(gs), dgl.batch(lgs), torch.stack(lats), torch.stack(ys), names


def run_element(element):
    print(f"\n==== Ablation: ALIGNN unfrozen backbone — {element} ====")

    out_dir = os.path.join(OUTPUT_ROOT, element)
    os.makedirs(out_dir, exist_ok=True)

    cif_dir = CIF_DIRS[element]
    id_prop_csv = os.path.join(cif_dir, CSV_NAMES[element])

    dataset = ALIGNNDataset(cif_dir, id_prop_csv)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_alignn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_alignn)

    model = get_figshare_model(MODEL_NAME)

    if FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    # else: full network stays trainable (ablation condition)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )
    criterion = torch.nn.L1Loss()  # MAE, matching the main study's ALIGNN loss

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for g, lg, lat, y, _ in train_loader:
            g, lg, lat, y = g.to(device), lg.to(device), lat.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model([g, lg, lat])
            loss = criterion(pred.view(-1), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{element}] Epoch {epoch+1}/{EPOCHS} Loss(MAE)={total_loss/len(train_loader):.6f}")

    torch.save(model.state_dict(), os.path.join(out_dir, f"alignn_unfrozen_{element}.pth"))

    model.eval()
    y_true_list, y_pred_list, names_list = [], [], []
    with torch.no_grad():
        for g, lg, lat, y, names in test_loader:
            g, lg, lat, y = g.to(device), lg.to(device), lat.to(device), y.to(device)
            pred = model([g, lg, lat])
            y_true_list.extend(y.view(-1).cpu().numpy().tolist())
            y_pred_list.extend(pred.view(-1).cpu().numpy().tolist())
            names_list.extend(names)

    y_true, y_pred = np.array(y_true_list), np.array(y_pred_list)

    df = pd.DataFrame({
        "file_name": names_list,
        "predicted_energy_per_atom": y_pred_list,
        "formation_energy_per_atom": y_true_list,
    })
    pred_csv = os.path.join(out_dir, f"predictions_{element}_unfrozen.csv")
    df.to_csv(pred_csv, index=False)

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    print(f"[{element}] UNFROZEN ABLATION RESULTS: MAE={mae:.4f}  MSE={mse:.4f}  (n_test={len(y_true)})")

    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)])
    plt.xlabel("True Formation Energy (eV/atom)")
    plt.ylabel("Predicted Formation Energy (eV/atom)")
    plt.title(f"ALIGNN Unfrozen-Backbone Ablation — {element}")
    plt.savefig(os.path.join(out_dir, f"scatter_{element}_unfrozen.png"), dpi=300)
    plt.close()

    return {"element": element, "mae": mae, "mse": mse, "n_test": len(y_true)}


if __name__ == "__main__":
    summary = [run_element(e) for e in ELEMENTS]
    summary_df = pd.DataFrame(summary)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary_df.to_csv(os.path.join(OUTPUT_ROOT, "summary_unfrozen_ablation.csv"), index=False)
    print("\n=== SUMMARY: ALIGNN Unfrozen-Backbone Ablation ===")
    print(summary_df.to_string(index=False))
