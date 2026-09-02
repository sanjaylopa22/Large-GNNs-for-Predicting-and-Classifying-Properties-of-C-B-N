
#!/usr/bin/env python
"""
ALIGNN Hybrid Random Sampling: 80/20 split per element (B, C, N)
- Training : 80% from each element, combined into one pool
- Testing  : Remaining 20% from each element, evaluated separately
- Fixed seed = 42 (single generator, consumed in order B → C → N)
- Saves per-element prediction CSVs, MAE/MSE metrics, and parity plots
"""

import os
import json
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig

# ============================================================
# SETTINGS
# ============================================================
MODEL_NAME = "jv_formation_energy_peratom_alignn"

ELEMENTS = ["B", "C", "N"]

CIF_DIRS = {
    "B": "./examples/sample_data/B_materials",
    "C": "./examples/sample_data/C_materials",
    "N": "./examples/sample_data/N_materials",
}

# CSV must have columns: file_name, formation_energy_per_atom
ID_PROP_CSVS = {
    "B": os.path.join(CIF_DIRS["B"], "id_prop_Boron.csv"),
    "C": os.path.join(CIF_DIRS["C"], "id_prop_Carbon.csv"),
    "N": os.path.join(CIF_DIRS["N"], "id_prop_Nitrogen.csv"),
}

OUTPUT_DIR    = "alignn_results_BCN_hybrid_claude"
OUTPUT_MODEL  = os.path.join(OUTPUT_DIR, "alignn_finetuned_BCN.pth")

BATCH_SIZE      = 4
EPOCHS          = 30
LEARNING_RATE   = 1e-4
CUTOFF          = 8.0
MAX_NEIGHBORS   = 12
FREEZE_BACKBONE = True   # only fine-tune the final FC head
RANDOM_SEED     = 42

device = torch.device("cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Element colours for plots
ELEM_COLORS = {"B": "#E05C2A", "C": "#2A7EC0", "N": "#2AB87A"}

# ============================================================
# PRETRAINED MODEL REGISTRY
# ============================================================
ALL_MODELS = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679",
        1,
    ]
}


def get_figshare_model(model_name: str) -> ALIGNN:
    """Download (once) and load the pretrained ALIGNN checkpoint."""
    url, _ = ALL_MODELS[model_name]
    zip_path = model_name + ".zip"

    if not os.path.exists(zip_path):
        print("Downloading pretrained ALIGNN model...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
        print("  Download complete.")

    with zipfile.ZipFile(zip_path) as z:
        chk_name = next(x for x in z.namelist() if "checkpoint_" in x)
        cfg_name  = next(x for x in z.namelist() if "config.json"  in x)

        config = json.loads(z.read(cfg_name))
        state_bytes = z.read(chk_name)

    model = ALIGNN(ALIGNNConfig(**config["model"]))

    # BUG FIX: close the fd returned by mkstemp to avoid descriptor leak
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    try:
        os.close(fd)                         # close the raw fd immediately
        with open(tmp_path, "wb") as f:
            f.write(state_bytes)
        ckpt = torch.load(tmp_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    model.to(device)
    print(f"  Pretrained ALIGNN loaded: {model_name}")
    return model


# ============================================================
# DATASET
# ============================================================
class ALIGNNDataset(Dataset):
    """
    Returns (g, lg, lattice, y, cif_name) for each structure.
    The cif_name is kept so predictions can be traced back to materials.
    """

    def __init__(self, cif_dir: str, id_prop_csv: str):
        self.cif_dir = cif_dir
        df = pd.read_csv(id_prop_csv)

        required = {"file_name", "formation_energy_per_atom"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{id_prop_csv} must have columns {required}. "
                f"Found: {list(df.columns)}"
            )

        self.data = df[["file_name", "formation_energy_per_atom"]].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cif_name, target = self.data[idx]

        if not cif_name.lower().endswith(".cif"):
            cif_name += ".cif"

        cif_path = os.path.join(self.cif_dir, cif_name)
        atoms    = Atoms.from_cif(cif_path)

        g, lg = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=CUTOFF,
            max_neighbors=MAX_NEIGHBORS,
        )

        lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
        y       = torch.tensor(float(target),     dtype=torch.float32)

        # BUG FIX: always return cif_name so collate and evaluation are consistent
        return g, lg, lattice, y, cif_name


# ============================================================
# COLLATE
# ============================================================
def collate_alignn(batch):
    """
    BUG FIX: unpack 5 items (g, lg, lat, y, name) — previously only 4.
    """
    import dgl

    gs, lgs, lats, ys, names = zip(*batch)

    return (
        dgl.batch(gs),
        dgl.batch(lgs),
        torch.stack(lats),
        torch.stack(ys),
        list(names),
    )


# ============================================================
# HYBRID 80/20 SPLIT  (single generator → deterministic order)
# ============================================================
def make_generator() -> torch.Generator:
    """Create a fresh seeded generator."""
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    return g


def create_hybrid_datasets():
    """
    80/20 split per element with a SINGLE generator consumed in
    element order (B → C → N).  The same generator must be recreated
    with make_generator() and consumed in the same order whenever the
    splits are reproduced (e.g. during per-element evaluation).

    Returns
    -------
    train_combined : ConcatDataset of 80% subsets for B, C, N
    test_per_elem  : dict {elem: Subset} for 20% subsets
    """
    generator = make_generator()

    train_datasets = []
    test_per_elem  = {}

    for elem in ELEMENTS:
        ds      = ALIGNNDataset(CIF_DIRS[elem], ID_PROP_CSVS[elem])
        n_total = len(ds)
        n_train = int(0.8 * n_total)
        n_test  = n_total - n_train

        # BUG FIX: reuse the same generator across elements (not re-seeded)
        train_set, test_set = random_split(ds, [n_train, n_test], generator=generator)

        print(f"  [{elem}]  total={n_total}  train={n_train}  test={n_test}")

        train_datasets.append(train_set)
        test_per_elem[elem] = test_set

    return ConcatDataset(train_datasets), test_per_elem


# ============================================================
# FINE-TUNING
# ============================================================
def finetune_BCN(model: ALIGNN, train_set) -> list:
    """Fine-tune model on the combined B+C+N training pool."""

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_alignn,
    )

    if FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            param.requires_grad = ("fc" in name)   # only unfreeze FC layers
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters (FC head only): {trainable:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    criterion = torch.nn.MSELoss()

    epoch_losses = []
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_batches  = 0

        for g, lg, lat, y, _ in train_loader:   # _ = names (not needed in training)
            g   = g.to(device)
            lg  = lg.to(device)
            lat = lat.to(device)
            y   = y.to(device)

            optimizer.zero_grad()
            pred = model([g, lg, lat]).reshape(-1)
            loss = criterion(pred, y.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"  Epoch [{epoch+1:03d}/{EPOCHS}]  Loss (MSE): {avg_loss:.6f}")

    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"\n  Fine-tuned model saved: {OUTPUT_MODEL}")

    return epoch_losses


# ============================================================
# EVALUATION (per element)
# ============================================================
def evaluate_per_element(model: ALIGNN, test_per_elem: dict) -> dict:
    """
    Run inference on each element's held-out 20%, compute MAE & MSE,
    save per-element CSV, and return results dict for plotting.
    """
    model.eval()
    results = {}

    for elem in ELEMENTS:
        test_set = test_per_elem[elem]
        print(f"\n[{elem}]  test samples = {len(test_set)}")

        test_loader = DataLoader(
            test_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_alignn,
        )

        y_true_list, y_pred_list, name_list = [], [], []

        with torch.no_grad():
            for g, lg, lat, y, names in test_loader:
                g   = g.to(device)
                lg  = lg.to(device)
                lat = lat.to(device)

                pred = model([g, lg, lat]).reshape(-1).cpu().numpy()

                y_true_list.extend(y.numpy().reshape(-1))
                y_pred_list.extend(pred)
                name_list.extend(names)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        mae  = float(np.mean(np.abs(y_true - y_pred)))
        mse  = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))

        print(f"  MAE  = {mae:.6f}  eV/atom")
        print(f"  MSE  = {mse:.6f}  (eV/atom)²")
        print(f"  RMSE = {rmse:.6f}  eV/atom")

        # Save CSV
        df_out = pd.DataFrame({
            "material"           : name_list,
            "ground_truth_eform" : y_true,
            "predicted_eform"    : y_pred,
            "error"              : y_pred - y_true,
            "abs_error"          : np.abs(y_pred - y_true),
        })
        csv_path = os.path.join(OUTPUT_DIR, f"ALIGNN_test_predictions_{elem}.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")

        results[elem] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "mae"   : mae,
            "mse"   : mse,
            "rmse"  : rmse,
            "n"     : len(y_true),
        }

    return results


# ============================================================
# PLOTTING
# ============================================================
def plot_loss_curve(epoch_losses: list):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, lw=2, color="#444")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("ALIGNN Fine-Tuning Loss — Combined B+C+N Training", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "finetune_loss_curve_BCN_claude.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Loss curve saved: {path}")


def plot_per_element_parity(results: dict):
    """One parity scatter plot per element."""
    for elem, res in results.items():
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        mae, mse, rmse = res["mae"], res["mse"], res["rmse"]
        color = ELEM_COLORS.get(elem, "#555")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.75, color=color,
                   edgecolors="k", linewidths=0.4, s=50, zorder=3)

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = 0.05 * (hi - lo)
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", lw=1.5, label="Ideal (y = x)", zorder=2)

        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True Formation Energy (eV/atom)", fontsize=12)
        ax.set_ylabel("Predicted Formation Energy (eV/atom)", fontsize=12)
        ax.set_title(
            f"ALIGNN Fine-Tuned — {elem} Test Set\n"
            f"MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  eV/atom",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = os.path.join(OUTPUT_DIR, f"parity_plot_{elem}.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"  Parity plot saved: {path}")


def plot_combined_parity(results: dict):
    """3-panel side-by-side parity plot for B, C, N."""
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    for idx, elem in enumerate(ELEMENTS):
        if elem not in results:
            continue
        res    = results[elem]
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        mae, mse, rmse = res["mae"], res["mse"], res["rmse"]
        color  = ELEM_COLORS.get(elem, "#555")

        ax = fig.add_subplot(gs[idx])
        ax.scatter(y_true, y_pred, alpha=0.75, color=color,
                   edgecolors="k", linewidths=0.3, s=40, zorder=3,
                   label=f"{elem}  (n={res['n']})")

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = 0.05 * (hi - lo)
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", lw=1.5, zorder=2)

        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"True $E_f$ (eV/atom)",      fontsize=11)
        ax.set_ylabel(r"Predicted $E_f$ (eV/atom)", fontsize=11)
        ax.set_title(
            f"Element: {elem}\n"
            f"MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}",
            fontsize=10,
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "ALIGNN Fine-Tuned: Parity Plots — B, C, N Test Sets\n"
        "(Hybrid 80/20 split per element, combined B+C+N training)",
        fontsize=13, y=1.02,
    )
    path = os.path.join(OUTPUT_DIR, "parity_plot_BCN_combined_claude.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined parity plot saved: {path}")


def save_summary(results: dict):
    rows = [
        {
            "Element"     : elem,
            "Test_Samples": res["n"],
            "MAE"         : round(res["mae"],  6),
            "MSE"         : round(res["mse"],  6),
            "RMSE"        : round(res["rmse"], 6),
        }
        for elem, res in results.items()
    ]
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "summary_metrics_BCN_claude.csv")
    df.to_csv(path, index=False)
    print(f"\nSummary metrics saved: {path}")
    print(df.to_string(index=False))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print(" STEP 1: Hybrid 80/20 per-element split (seed=42)")
    print("=" * 60)
    train_set, test_per_elem = create_hybrid_datasets()
    print(f"\n  Combined training pool : {len(train_set)} structures")

    print("\n" + "=" * 60)
    print(" STEP 2: Load pretrained ALIGNN")
    print("=" * 60)
    model = get_figshare_model(MODEL_NAME)

    print("\n" + "=" * 60)
    print(f" STEP 3: Fine-tune  (epochs={EPOCHS}, lr={LEARNING_RATE}, "
          f"batch={BATCH_SIZE}, freeze_backbone={FREEZE_BACKBONE})")
    print("=" * 60)
    epoch_losses = finetune_BCN(model, train_set)
    plot_loss_curve(epoch_losses)

    print("\n" + "=" * 60)
    print(" STEP 4: Per-element evaluation on held-out 20%")
    print("=" * 60)
    results = evaluate_per_element(model, test_per_elem)

    print("\n" + "=" * 60)
    print(" STEP 5: Saving plots and summary")
    print("=" * 60)
    plot_per_element_parity(results)
    plot_combined_parity(results)
    save_summary(results)

    print(f"\nAll outputs written to: {os.path.abspath(OUTPUT_DIR)}")

'''
import os
import json
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig


# ---------------- SETTINGS ----------------

MODEL_NAME = "jv_formation_energy_peratom_alignn"

ELEMENTS = ["B", "C", "N"]

CIF_DIRS = {
    "B": "./examples/sample_data/B_materials",
    "C": "./examples/sample_data/C_materials",
    "N": "./examples/sample_data/N_materials",
}

ID_PROP_CSVS = {
    "B": os.path.join(CIF_DIRS["B"], "id_prop_Boron.csv"),
    "C": os.path.join(CIF_DIRS["C"], "id_prop_Carbon.csv"),
    "N": os.path.join(CIF_DIRS["N"], "id_prop_Nitrogen.csv"),
}

OUTPUT_MODEL = "alignn_finetuned_BCN.pth"

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
CUTOFF = 8.0
MAX_NEIGHBORS = 12
FREEZE_BACKBONE = True

RANDOM_SEED = 42

device = torch.device("cpu")


# ---------------- FIGSHARE MODEL ----------------

ALL_MODELS = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679",
        1,
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


# ---------------- DATASET ----------------

class ALIGNNDataset(Dataset):

    def __init__(self, cif_dir, id_prop_csv):

        self.cif_dir = cif_dir

        df = pd.read_csv(id_prop_csv)

        self.data = df[
            ["file_name", "formation_energy_per_atom"]
        ].values.tolist()

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        cif_name, target = self.data[idx]

        if not cif_name.endswith(".cif"):
            cif_name += ".cif"

        path = os.path.join(
            self.cif_dir,
            cif_name,
        )

        atoms = Atoms.from_cif(path)

        g, lg = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=CUTOFF,
            max_neighbors=MAX_NEIGHBORS,
        )

        lattice = torch.tensor(
            atoms.lattice_mat,
            dtype=torch.float32,
        )

        y = torch.tensor(
            float(target),
            dtype=torch.float32,
        )

        return g, lg, lattice, y, cif_name


# ---------------- COLLATE ----------------

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


# ---------------- HYBRID SPLIT ----------------

def create_hybrid_datasets():

    train_datasets = []
    test_datasets = []

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    for elem in ELEMENTS:

        ds = ALIGNNDataset(
            CIF_DIRS[elem],
            ID_PROP_CSVS[elem],
        )

        n_total = len(ds)

        n_train = int(0.8 * n_total)

        n_test = n_total - n_train

        train_set, test_set = random_split(
            ds,
            [n_train, n_test],
            generator=generator,
        )

        train_datasets.append(train_set)

        test_datasets.append(test_set)

    from torch.utils.data import ConcatDataset

    train_hybrid = ConcatDataset(train_datasets)

    test_hybrid = ConcatDataset(test_datasets)

    return train_hybrid, test_hybrid


# ---------------- FINETUNE ----------------

def finetune_BCN():

    print("\n" + "=" * 60)
    print("Fine-tuning ALIGNN on B/C/N")
    print("=" * 60)

    train_set, _ = create_hybrid_datasets()

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_alignn,
    )

    model = get_figshare_model(
        MODEL_NAME
    )

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

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0

        for g, lg, lat, y, _ in train_loader:

            g = g.to(device)
            lg = lg.to(device)
            lat = lat.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model([g, lg, lat]).reshape(-1)

            loss = criterion(
                pred,
                y.reshape(-1),
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

    evaluate_per_element(model)


# ---------------- TEST ----------------

def evaluate_per_element(model):

    model.eval()

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    for elem in ELEMENTS:

        print("\n" + "=" * 60)
        print("Testing:", elem)
        print("=" * 60)

        dataset = ALIGNNDataset(
            CIF_DIRS[elem],
            ID_PROP_CSVS[elem],
        )

        n_total = len(dataset)

        n_train = int(0.8 * n_total)

        n_test = n_total - n_train

        _, test_subset = random_split(
            dataset,
            [n_train, n_test],
            generator=generator,
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_alignn,
        )

        y_true = []
        y_pred = []
        names_list = []

        with torch.no_grad():

            for g, lg, lat, y, names in test_loader:

                g = g.to(device)
                lg = lg.to(device)
                lat = lat.to(device)

                pred = model(
                    [g, lg, lat]
                ).reshape(-1)

                y_true.extend(
                    y.numpy().reshape(-1)
                )

                y_pred.extend(
                    pred.cpu().numpy().reshape(-1)
                )

                names_list.extend(
                    names
                )

        y_true = np.array(y_true)

        y_pred = np.array(y_pred)

        mae = np.mean(
            np.abs(y_true - y_pred)
        )

        mse = np.mean(
            (y_true - y_pred) ** 2
        )

        print("Test:", len(test_subset))
        print("MAE:", mae)
        print("MSE:", mse)

        df = pd.DataFrame(
            {
                "material": names_list,
                "ground_truth_eform": y_true,
                "predicted_eform": y_pred,
            }
        )

        save_name = (
            f"ALIGNN_test_predictions_{elem}.csv"
        )

        df.to_csv(
            save_name,
            index=False,
        )

        print("Saved:", save_name)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    finetune_BCN()


'''