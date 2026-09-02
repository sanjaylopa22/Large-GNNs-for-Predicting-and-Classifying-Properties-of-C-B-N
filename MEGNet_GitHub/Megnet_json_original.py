# MEGNet finetune on MP dataset we use a train-validation-test split of 60,000–5000–4239

import json
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from megnet.utils.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import tensorflow as tf

# ===============================
# USER CONFIG
# ===============================
JSON_PATH      = "mp.2018.6.1.json"
OUTPUT_CSV     = "MEGNet_test_results_predict_original.csv"
OUTPUT_VAL_CSV = "MEGNet_val_results_predict_original.csv"

# Split sizes  (must sum to ≤ total dataset length)
N_TRAIN = 60000
N_VAL   = 5000
N_TEST  = 4239   # indices [65000 : 69239]

# Fine-tuning hyper-parameters
EPOCHS        = 50
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4
SEED          = 42

# ===============================
# LOAD JSON DATA
# ===============================
print("Loading MP JSON dataset...")
with open(JSON_PATH, "r") as f:
    mp_data = json.load(f)
print(f"Total structures in JSON : {len(mp_data)}")

# -------------------------------------------------------
# Deterministic splits  (no shuffling — index-based)
# Train : [0       : 60000]
# Val   : [60000   : 65000]
# Test  : [65000   : 69239]
# -------------------------------------------------------
train_data = mp_data[0       : N_TRAIN]
val_data   = mp_data[N_TRAIN : N_TRAIN + N_VAL]
test_data  = mp_data[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]

print(f"Train : {len(train_data):>6} structures")
print(f"Val   : {len(val_data):>6} structures")
print(f"Test  : {len(test_data):>6} structures")
print(f"Total : {len(train_data)+len(val_data)+len(test_data):>6} structures\n")

# ===============================
# HELPER: JSON entry → pymatgen Structure
# ===============================
def parse_structure(entry):
    struct_entry = entry.get("structure", None)
    mid = entry.get("material_id", "unknown")

    if not struct_entry:
        raise ValueError(f"Empty/missing 'structure' for entry: {mid}")

    if isinstance(struct_entry, dict):
        return Structure.from_dict(struct_entry)

    if isinstance(struct_entry, str):
        s = struct_entry.strip()
        if not s:
            raise ValueError(f"Empty structure string for entry: {mid}")
        if s.startswith("#") or "_cell_length" in s:
            return Structure.from_str(s, fmt="cif")
        return Structure.from_dict(json.loads(s))

    raise ValueError(f"Unrecognised structure type {type(struct_entry)} for entry: {mid}")


# ===============================
# HELPER: Load structures from a split
# ===============================
def load_split(split_data, label=""):
    structures, targets, ids = [], [], []
    skipped = 0

    for d in split_data:
        mid = d.get("material_id", "unknown")
        try:
            s  = parse_structure(d)
            fe = float(d["formation_energy_per_atom"])
            structures.append(s)
            targets.append(fe)
            ids.append(mid)
        except Exception as e:
            print(f"  [{label}] Skipping {mid}: {e}")
            skipped += 1

    print(f"  [{label}] Loaded {len(structures)} / {len(split_data)}  (skipped={skipped})")
    return structures, np.array(targets), ids


# ===============================
# PARSE ALL THREE SPLITS
# ===============================
print("Parsing structures...")
train_structures, train_targets, train_ids = load_split(train_data, "TRAIN")
val_structures,   val_targets,   val_ids   = load_split(val_data,   "VAL")
test_structures,  test_targets,  test_ids  = load_split(test_data,  "TEST")

for label, structs in [("Train", train_structures),
                        ("Val",   val_structures),
                        ("Test",  test_structures)]:
    if len(structs) == 0:
        raise RuntimeError(
            f"No valid structures parsed for {label} split.\n"
            "Check key names: 'material_id', 'structure', 'formation_energy_per_atom'"
        )

# ===============================
# LOAD PRETRAINED MEGNet MODEL
# ===============================
print("\nLoading pretrained MEGNet model...")
model = load_model("Eform_MP_2018")
print("Model loaded.")

# ===============================
# RECOMPILE FOR FINE-TUNING
# ===============================
print(f"\nRecompiling with Adam lr={LEARNING_RATE}...")
model.model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mae",
)

# ===============================
# FINE-TUNE ON TRAINING SET
# ===============================
print(f"\nFine-tuning: epochs={EPOCHS}, batch_size={BATCH_SIZE} ...")
history = model.train(
    train_structures,
    train_targets.tolist(),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    scrub_failed_structures=True,
)
print("Fine-tuning complete.\n")

# -----------------------------------------------
# Plot training loss curve
# -----------------------------------------------
if history is not None:
    try:
        loss_key = next(k for k in history.history if "loss" in k.lower())
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.history[loss_key], lw=2, label="Train MAE")
        val_key = "val_" + loss_key
        if val_key in history.history:
            ax.plot(history.history[val_key], lw=2, linestyle="--", label="Val MAE")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MAE Loss (eV/atom)", fontsize=12)
        ax.set_title("MEGNet Fine-Tuning Loss — MP 2018", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("finetune_loss_curve_MP.png", dpi=300)
        plt.close(fig)
        print("Loss curve saved: finetune_loss_curve_MP.png")
    except Exception as e:
        print(f"[INFO] Could not plot loss curve: {e}")


# ===============================
# PREDICTION HELPER
# ===============================
def predict_set(structures, ids, label=""):
    preds, failed_idx = [], []

    for i, s in enumerate(structures):
        try:
            p = float(model.predict_structure(s).ravel()[0])
            preds.append(p)
        except Exception as e:
            print(f"  [WARN] {label} prediction failed for {ids[i]} (index {i}): {e}")
            failed_idx.append(i)

    return np.array(preds), failed_idx


def drop_failed(targets, ids, failed_idx):
    if not failed_idx:
        return targets, ids
    mask = np.ones(len(targets), dtype=bool)
    mask[failed_idx] = False
    return targets[mask], [v for i, v in enumerate(ids) if i not in set(failed_idx)]


# ===============================
# VALIDATION SET PREDICTIONS
# ===============================
print("Predicting on validation set...")
val_preds, val_failed = predict_set(val_structures, val_ids, label="VAL")
val_targets_clean, val_ids_clean = drop_failed(val_targets, val_ids, val_failed)
print(f"  Val predictions: {len(val_preds)} succeeded, {len(val_failed)} failed.")

val_mae = mean_absolute_error(val_targets_clean, val_preds)
val_mse = mean_squared_error(val_targets_clean, val_preds)

print(f"\n--- Validation Metrics ---")
print(f"  Samples : {len(val_preds)}")
print(f"  MAE     : {val_mae:.4f} eV/atom")
print(f"  MSE     : {val_mse:.4f} (eV/atom)²")

val_df = pd.DataFrame({
    "id"        : val_ids_clean,
    "true_eform": val_targets_clean,
    "pred_eform": val_preds,
    "error"     : val_preds - val_targets_clean,
})
val_df.to_csv(OUTPUT_VAL_CSV, index=False)
print(f"  Val results saved: {OUTPUT_VAL_CSV}")

# ===============================
# TEST SET PREDICTIONS
# ===============================
print("\nPredicting on test set...")
test_preds, test_failed = predict_set(test_structures, test_ids, label="TEST")
test_targets_clean, test_ids_clean = drop_failed(test_targets, test_ids, test_failed)
print(f"  Test predictions: {len(test_preds)} succeeded, {len(test_failed)} failed.")

if len(test_preds) == 0:
    raise RuntimeError(
        "All test predictions failed. See [WARN] lines above.\n"
        "Common causes: structures too small for cutoff, or graph-build errors."
    )

test_mae = mean_absolute_error(test_targets_clean, test_preds)
test_mse = mean_squared_error(test_targets_clean, test_preds)

print(f"\n--- Test Metrics ---")
print(f"  Samples : {len(test_preds)}")
print(f"  MAE     : {test_mae:.4f} eV/atom")
print(f"  MSE     : {test_mse:.4f} (eV/atom)²")

results_df = pd.DataFrame({
    "id"        : test_ids_clean,
    "true_eform": test_targets_clean,
    "pred_eform": test_preds,
    "error"     : test_preds - test_targets_clean,
})
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"  Test results saved: {OUTPUT_CSV}")

# ===============================
# PARITY PLOTS  (val + test, side-by-side)
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, y_true, y_pred, mae, mse, label, color in [
    (axes[0], val_targets_clean,  val_preds,
     val_mae,  val_mse,  "Validation (5 000)", "#2A7EC0"),
    (axes[1], test_targets_clean, test_preds,
     test_mae, test_mse, "Test (4 239)",        "#E05C2A"),
]:
    ax.scatter(y_true, y_pred, alpha=0.5, s=8, color=color,
               edgecolors="none", zorder=3)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    margin = 0.05 * (hi - lo)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", lw=1.5, label="Ideal", zorder=2)

    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True Formation Energy (eV/atom)", fontsize=11)
    ax.set_ylabel("Predicted Formation Energy (eV/atom)", fontsize=11)
    ax.set_title(
        f"MEGNet — {label}\nMAE={mae:.4f}  MSE={mse:.4f}  eV/atom",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "MEGNet Parity Plots — MP 2018\n"
    "Train=60 000 | Val=5 000 | Test=4 239",
    fontsize=13,
)
fig.tight_layout()
fig.savefig("Formation_energy_comparison_Megnet_predict_original.png", dpi=300)
plt.close(fig)
print("\nParity plot saved: Formation_energy_comparison_Megnet_predict_original.png")

# ===============================
# FINAL SUMMARY
# ===============================
print("\n" + "=" * 45)
print("  MEGNet MP 2018 — Final Summary")
print("=" * 45)
print(f"  Train samples : {len(train_structures)}")
print(f"  Val   samples : {len(val_preds)}")
print(f"  Test  samples : {len(test_preds)}")
print(f"  Val  MAE      : {val_mae:.4f}  eV/atom")
print(f"  Val  MSE      : {val_mse:.4f}  (eV/atom)²")
print(f"  Test MAE      : {test_mae:.4f}  eV/atom")
print(f"  Test MSE      : {test_mse:.4f}  (eV/atom)²")
print("=" * 45)

'''
import json
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from megnet.utils.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# ===============================
# USER CONFIG
# ===============================
JSON_PATH = "mp.2018.6.1.json"
OUTPUT_CSV = "MP_test_results_predict_original.csv"
CUTOFF = 4.0  # angstroms

# ===============================
# LOAD JSON DATA
# ===============================
print("Loading MP JSON dataset...")
with open(JSON_PATH, "r") as f:
    mp_data = json.load(f)
print(f"Total structures in JSON: {len(mp_data)}")

# Test split
test_data = mp_data[65000:69239]
print(f"Test set: {len(test_data)} structures")

# ===============================
# HELPER: convert JSON entry to pymatgen Structure
# Handles CIF strings, pymatgen JSON dicts, and JSON-encoded dict strings
# ===============================
def parse_structure(entry):
    struct_entry = entry.get("structure", None)
    mid = entry.get("material_id", "unknown")

    if not struct_entry:
        raise ValueError(f"Empty/missing 'structure' for entry: {mid}")

    if isinstance(struct_entry, dict):
        # Already a pymatgen-style dict
        return Structure.from_dict(struct_entry)

    if isinstance(struct_entry, str):
        s = struct_entry.strip()
        if not s:
            raise ValueError(f"Empty structure string for entry: {mid}")
        # CIF format — starts with a '#' comment or contains CIF keywords
        if s.startswith("#") or "_cell_length" in s:
            return Structure.from_str(s, fmt="cif")
        # Otherwise assume it is a JSON-encoded pymatgen dict
        return Structure.from_dict(json.loads(s))

    raise ValueError(f"Unrecognised structure type {type(struct_entry)} for entry: {mid}")


# ===============================
# PARSE TEST STRUCTURES
# ===============================
print("Converting test structures...")
test_structures = []
test_targets = []
valid_ids = []

for d in test_data:
    mid = d.get("material_id", "unknown")
    try:
        s = parse_structure(d)
        # Correct key is 'formation_energy_per_atom', not 'formation_energy'
        fe = float(d["formation_energy_per_atom"])
        test_structures.append(s)
        test_targets.append(fe)
        valid_ids.append(mid)
    except Exception as e:
        print(f"  Skipping {mid}: {e}")

test_targets = np.array(test_targets)
print(f"Loaded {len(test_structures)} valid test structures.")

if len(test_structures) == 0:
    raise RuntimeError(
        "No valid structures were parsed.\n"
        "Check the key names in your JSON — this dataset uses:\n"
        "  'material_id', 'structure' (CIF string), 'formation_energy_per_atom'"
    )

# ===============================
# LOAD PRETRAINED MEGNet MODEL
# ===============================
print("\nLoading pretrained MEGNet model...")
pretrained_model = load_model("Eform_MP_2018")
print("Model loaded.")

# ===============================
# PREDICTION ON TEST SET
# ===============================
print("\nPredicting formation energies for test set...")
preds = []
failed_indices = []

for i, s in enumerate(test_structures):
    try:
        pred_e = pretrained_model.predict_structure(s).ravel()[0]
        preds.append(pred_e)
    except Exception as e:
        print(f"  [WARN] Prediction failed for {valid_ids[i]} (index {i}): {e}")
        failed_indices.append(i)

# Drop failed entries from targets/ids so arrays stay aligned
if failed_indices:
    mask = np.ones(len(test_targets), dtype=bool)
    mask[failed_indices] = False
    test_targets = test_targets[mask]
    valid_ids = [v for i, v in enumerate(valid_ids) if i not in set(failed_indices)]

preds = np.array(preds)
print(f"Predictions completed: {len(preds)} succeeded, {len(failed_indices)} failed.")

if len(preds) == 0:
    raise RuntimeError(
        "All predictions failed. See [WARN] lines above for the root cause.\n"
        "Common causes: structures too small for the cutoff radius, or graph-build errors."
    )

# ===============================
# METRICS
# ===============================
mae = mean_absolute_error(test_targets, preds)
mse = mean_squared_error(test_targets, preds)

print("\n===============================")
print("MEGNet Formation Energy Results on Test Set")
print("===============================")
print(f"Number of materials : {len(preds)}")
print(f"MAE                 : {mae:.4f} eV/atom")
print(f"MSE                 : {mse:.4f} (eV/atom)^2")

# ===============================
# SAVE RESULTS
# ===============================
results_df = pd.DataFrame({
    "id":         valid_ids,
    "true_eform": test_targets,
    "pred_eform": preds,
    "error":      preds - test_targets,
})
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to: {OUTPUT_CSV}")

# ===============================
# PARITY PLOT
# ===============================
plt.figure(figsize=(6, 6))
plt.scatter(test_targets, preds, alpha=0.7)
min_val = min(test_targets.min(), preds.min())
max_val = max(test_targets.max(), preds.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--")
plt.xlabel("True Formation Energy (eV/atom)")
plt.ylabel("Predicted Formation Energy (eV/atom)")
plt.title("MEGNet Formation Energy Prediction (MP 2018, No Fine-Tuning)")
plt.tight_layout()
plt.savefig("Formation_energy_comparison_Megnet_predict_original.png", dpi=300)
plt.close()
print("Parity plot saved: Formation_energy_comparison_Megnet_predict_original.png")
'''