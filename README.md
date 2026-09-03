# Large-Scale Pretrained and Fine-Tuned GNNs for Predicting and Classifying Properties of Carbon–Boron–Nitrogen Materials

Benchmarking framework for formation-energy regression and thermodynamic-stability classification of Boron (B), Carbon (C), and Nitrogen (N) crystal structures curated from the [Materials Project](https://materialsproject.org) database, using three graph neural network architectures — **CGCNN**, **ALIGNN**, and **MEGNet** — under pretrained (zero-shot), dataset-specific fine-tuned (DS–B/C/N), and hybrid fine-tuned (Hybrid BCN) regimes.

---

## Overview

This project evaluates whether large pretrained GNN property predictors transfer effectively to chemically narrow, element-constrained materials subspaces, and whether the resulting formation-energy predictions are reliable enough to drive downstream thermodynamic stability screening.

Two questions are addressed:

1. **Regression**: How accurately can CGCNN, ALIGNN, and MEGNet predict formation energy per atom ($E_f$, eV/atom) for B, C, and N compounds, before and after fine-tuning?
2. **Classification**: When predicted $E_f$ is combined with the Materials Project's energy-above-hull ($E_\text{hull}$) values, how reliably can a model classify a compound as thermodynamically **Stable** or **Unstable**?

---

## Key Contributions

- **Element-specific dataset curation** — Boron, Carbon, and Nitrogen subsets extracted from the Materials Project, each stored as paired `.cif` structure files and an `id_prop_*.csv` of formation energies.
- **Systematic cross-model, cross-element benchmarking** of CGCNN, ALIGNN, and MEGNet under a common evaluation protocol (fixed 80/20 splits, seed = 42, Adam optimizer, matched learning rate).
- **Dual fine-tuning strategy** — dataset-specific (DS–B/C/N) and hybrid (Hybrid BCN) fine-tuning, implemented for all three architectures.
- **Thermodynamic stability classification** using the joint criterion $\hat{E}_f < 0$ **and** $\hat{E}_f \leq E_\text{hull}$, evaluated via accuracy, precision, recall, and F1-score against DFT ground truth.
- **Reproducible protocol** — fixed random seeds, standardized train/test splits, and documented (not artificially equalized) model-specific fine-tuning settings.

---

## Repository Structure

```
.
├── data/
│   ├── B_materials/                # Boron .cif files + id_prop_Boron.csv
│   ├── C_materials/                # Carbon .cif files + id_prop_Carbon.csv
│   └── N_materials/                # Nitrogen .cif files + id_prop_Nitrogen.csv
│
├── pretrained_inference/
│   ├── batch_predict_alignn_B.py
│   ├── batch_predict_alignn_C.py
│   ├── batch_predict_alignn_N.py
│   ├── Megnet_B.py
│   ├── Megnet_C.py
│   └── Megnet_N.py
│
├── finetuning/
│   ├── finetuned_alignn_B.py
│   ├── finetuned_alignn_C.py
│   ├── finetuned_alignn_N.py
│   ├── finetuned_ALIGNN_BCN_hybrid_random.py
│   ├── predict_finetune_CGCNN_B.py
│   ├── predict_finetune_CGCNN_C.py
│   ├── predict_finetune_CGCNN_N.py
│   ├── predict_finetune_CGCNN_BCN_hybrid.py
│   ├── Megnet_Finetune_random_B.py
│   ├── Megnet_Finetune_random_C.py
│   ├── Megnet_Finetune_random_N.py
│   └── Megnet_Finetune_hybrid_BCN.py
│
├── stability_classification/
│   ├── stability_classification_MEGNet.py
│   ├── stability_classification_CGCNN.py
│   └── stability_classification_ALIGNN.py
│
├── outputs/
│   ├── stability_results_MEGNet/
│   ├── stability_results_CGCNN/
│   ├── stability_results_ALIGNN/
│   └── (per-model prediction CSVs, parity/scatter plots, loss curves)
│
└── README.md
```

---

## Requirements

```bash
pip install torch dgl pymatgen scikit-learn pandas numpy matplotlib
pip install cgcnn        # or clone https://github.com/txie-93/cgcnn
pip install alignn       # https://github.com/usnistgov/alignn
pip install megnet       # https://github.com/materialsvirtuallab/megnet
pip install mp-api       # Materials Project API client, for E_hull fetching
```

A [Materials Project API key](https://next-gen.materialsproject.org/api) is required for both dataset curation and $E_\text{hull}$ retrieval. Set it as an environment variable rather than hardcoding it in any script:

```bash
export MP_API_KEY="your_api_key_here"
```

> 🔒 **Security note:** Never commit an API key directly into a script's source code. All scripts in this repo read `MP_API_KEY` from the environment, falling back to a placeholder if unset.

---

## Usage

### 1. Pretrained (zero-shot) baseline inference

```bash
python pretrained_inference/batch_predict_alignn_B.py
python pretrained_inference/Megnet_B.py
# CGCNN pretrained baseline uses the same 80/20 split inside predict_finetune_CGCNN_*.py
# (see the "Fine-tuning" step below; pretrained metrics are reported before the fine-tune loop runs)
```

### 2. Fine-tuning (dataset-specific and hybrid)

```bash
# Dataset-specific (per element)
python finetuning/finetuned_alignn_B.py
python finetuning/Megnet_Finetune_random_B.py
python finetuning/predict_finetune_CGCNN_B.py <pretrained_model_path> data/B_materials

# Hybrid (jointly trained on B+C+N, evaluated per element)
python finetuning/finetuned_ALIGNN_BCN_hybrid_random.py
python finetuning/Megnet_Finetune_hybrid_BCN.py
python finetuning/predict_finetune_CGCNN_BCN_hybrid.py <pretrained_model_path> \
    --b-path data/B_materials --c-path data/C_materials --n-path data/N_materials
```

Each script performs a fixed 80/20 train-test split (`random_state=42` / `torch.Generator().manual_seed(42)`), fine-tunes for a fixed number of epochs (see [Hyperparameters](#hyperparameters) below — **no early stopping or validation split is used**), and saves per-structure prediction CSVs and parity/scatter plots.

### 3. Thermodynamic stability classification

Once prediction CSVs exist for all three elements and a given model:

```bash
python stability_classification/stability_classification_MEGNet.py
python stability_classification/stability_classification_CGCNN.py
python stability_classification/stability_classification_ALIGNN.py
```

Each script:
1. Fetches $E_\text{hull}$ for every material ID via the Materials Project API.
2. Classifies each structure as **Stable** (`E < 0` and `E ≤ E_hull`) or **Unstable**, using both the predicted and ground-truth formation energy.
3. Computes Accuracy, Precision, Recall, and F1-score (positive class = "Stable"), per element and pooled across B+C+N.
4. Saves per-structure labeled CSVs, confusion-matrix plots, stability-distribution plots, and a summary metrics table to `outputs/stability_results_<MODEL>/`.

**Before running**, update the `PRED_CSV_PATHS` dictionary in each stability script to point at your actual prediction CSV filenames — these vary by model (see [Output columns](#output-columns) below) and by which fine-tuning run you want to evaluate.

---

## Hyperparameters

| Hyperparameter | CGCNN | ALIGNN | MEGNet |
|---|---|---|---|
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| Batch size | 64 | 4 | 64 |
| Epochs | 50 | 30 | 50 |
| Optimizer | Adam | Adam | Adam |
| Loss function | MSE | MSE | MAE |
| Train/test split | 80/20 (seed 42) | 80/20 (seed 42) | 80/20 (seed 42) |
| Backbone frozen | No | Yes (output head only trainable) | No |
| Validation split | None | None | None |
| Early stopping | No (fixed epoch count) | No (fixed epoch count) | No (fixed epoch count) |

> These settings are **documented, not equalized**, across architectures. In particular, ALIGNN uses a frozen backbone and MSE loss where CGCNN and MEGNet fine-tune the full network — this is a known confound when comparing cross-model results and is discussed in the accompanying paper.

---

## Output columns

Prediction CSVs use different column names depending on the model — the stability classification scripts account for this, but keep it in mind if writing your own downstream analysis:

| Model | Prediction CSV columns |
|---|---|
| MEGNet | `id`, `true_eform`, `pred_eform` |
| CGCNN | `Material`, `Target_FE`, `Predicted_FE` |
| ALIGNN | `file_name`, `predicted_energy_per_atom`, `formation_energy_per_atom` |

---

## Results Summary

Fine-tuning substantially improves formation-energy prediction over pretrained baselines across all three architectures and all three elements. MEGNet achieves the lowest regression error on Boron (MSE = 0.0099 (eV/atom)², MAE = 0.0420 eV/atom after dataset-specific fine-tuning — a ~78%/~50% reduction over the pretrained baseline) and the lowest MAE across all three elements under hybrid fine-tuning. CGCNN achieves the lowest MSE on Carbon and Nitrogen under hybrid fine-tuning. Downstream stability-classification accuracy tracks regression accuracy closely across all models; pooled across B, C, and N, MEGNet achieves the highest combined accuracy (0.973) and F1-score (0.983), followed by CGCNN (0.955 / 0.972) and ALIGNN (0.943 / 0.965).

Full results tables, confusion matrices, and per-element analysis are provided in the accompanying paper (see [Citation](#citation)).

---

## Citation

If you use this code or benchmarking framework, please cite the original architecture papers alongside this work:

```bibtex
@article{jain2013materials,
  author  = {Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin A.},
  title   = {Commentary: The {Materials Project}: A materials genome approach to accelerating materials innovation},
  journal = {APL Materials},
  year    = {2013}, volume = {1}, number = {1}, pages = {011002},
  doi     = {10.1063/1.4812323}
}

@article{xie2018cgcnn,
  author  = {Xie, Tian and Grossman, Jeffrey C.},
  title   = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  journal = {Physical Review Letters},
  year    = {2018}, volume = {120}, number = {14}, pages = {145301},
  doi     = {10.1103/PhysRevLett.120.145301}
}

@article{chen2019megnet,
  author  = {Chen, Chi and Ye, Weike and Zuo, Yunxing and Zheng, Chen and Ong, Shyue Ping},
  title   = {Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals},
  journal = {Chemistry of Materials},
  year    = {2019}, volume = {31}, number = {9}, pages = {3564--3572},
  doi     = {10.1021/acs.chemmater.9b01294}
}

@article{choudhary2021alignn,
  author  = {Choudhary, Kamal and DeCost, Brian},
  title   = {Atomistic Line Graph Neural Network for improved materials property predictions},
  journal = {npj Computational Materials},
  year    = {2021}, volume = {7}, pages = {185},
  doi     = {10.1038/s41524-021-00650-1}
}
```

---

## License

Add your chosen license here (e.g., MIT, Apache-2.0) — none was specified in the source material for this README.

## Contact

Sanjay Chakraborty — [ResearchGate](https://www.researchgate.net/profile/Sanjay_Chakraborty) · [Google Scholar](https://scholar.google.co.in/citations?user=vtUt3S4AAAAJ&hl=en)
