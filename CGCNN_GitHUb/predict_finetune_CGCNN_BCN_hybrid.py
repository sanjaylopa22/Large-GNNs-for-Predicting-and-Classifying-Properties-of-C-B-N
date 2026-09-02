# A hybrid random sampling approach for fine-tuning/predicting formation energy, where each element (B, C, N) contributes 80% of its own dataset to the training set and 20% to the test set, and then the training sets from all elements are merged, 
# and the same for test sets. This ensures stratified per-element splitting.

import argparse
import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt

from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

# --------------------------------------------------
# Argument Parser
# --------------------------------------------------
parser = argparse.ArgumentParser(description='CGCNN Hybrid Fine-Tuning (B/C/N)')
parser.add_argument('modelpath', help='Path to pretrained model')
parser.add_argument('--b-path', required=True, help='Path to Boron CIFs')
parser.add_argument('--c-path', required=True, help='Path to Carbon CIFs')
parser.add_argument('--n-path', required=True, help='Path to Nitrogen CIFs')
parser.add_argument('-b', '--batch-size', default=64, type=int)
parser.add_argument('-e', '--epochs', default=50, type=int)
parser.add_argument('--disable-cuda', action='store_true')

args = parser.parse_args(sys.argv[1:])
args.cuda = not args.disable_cuda and torch.cuda.is_available()

# --------------------------------------------------
# Normalizer
# --------------------------------------------------
class Normalizer(object):
    def __init__(self):
        self.mean = 0
        self.std = 1

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


# --------------------------------------------------
# Dataset split
# --------------------------------------------------
def element_split(dataset, train_ratio=0.8, seed=42):
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_test = n_total - n_train
    train_set, test_set = random_split(dataset,
                                       [n_train, n_test],
                                       generator=torch.Generator().manual_seed(seed))
    return train_set, test_set

def load_cif_dataset(cif_path):
    dataset = CIFData(cif_path)
    print(f"{len(dataset)} samples found in {cif_path}")
    return dataset

# --------------------------------------------------
# Evaluation function
# --------------------------------------------------
def evaluate(model, normalizer, test_dataset, element_name):
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate_pool)

    model.eval()
    test_targets, test_preds, test_ids = [], [], []

    with torch.no_grad():
        for input, target, batch_cif_ids in test_loader:
            if args.cuda:
                input_var = (input[0].cuda(),
                             input[1].cuda(),
                             input[2].cuda(),
                             [idx.cuda() for idx in input[3]])
                target = target.cuda()
            else:
                input_var = input

            output = model(*input_var)
            denorm_output = normalizer.denorm(output.cpu())
            target_cpu = target.cpu()

            test_preds += denorm_output.view(-1).tolist()
            test_targets += target_cpu.view(-1).tolist()
            test_ids += batch_cif_ids

    test_targets_arr = np.array(test_targets)
    test_preds_arr = np.array(test_preds)

    mae_value = np.mean(np.abs(test_targets_arr - test_preds_arr))
    mse_value = np.mean((test_targets_arr - test_preds_arr) ** 2)

    print(f"\n=== TEST RESULTS: {element_name} ===")
    print(f"MAE: {mae_value:.6f} eV/atom")
    print(f"MSE: {mse_value:.6f} (eV/atom)^2")

    # Save CSV
    csv_file = f'test_results_{element_name}.csv'
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Material', 'Target_FE', 'Predicted_FE'])
        for cif_id, target, pred in zip(test_ids, test_targets_arr, test_preds_arr):
            formula = os.path.splitext(os.path.basename(cif_id))[0]
            writer.writerow((formula, target, pred))

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(test_targets_arr, test_preds_arr)
    plt.plot([min(test_targets_arr), max(test_targets_arr)],
             [min(test_targets_arr), max(test_targets_arr)], 'r--')
    plt.xlabel("Ground Truth Formation Energy (eV/atom)")
    plt.ylabel("Predicted Formation Energy (eV/atom)")
    plt.title(f"CGCNN Fine-Tuned Model ({element_name})")
    plt.grid(True)
    plt.savefig(f"{element_name}_scatter.png", dpi=300)
    plt.close()

    print(f"Saved: {csv_file}, {element_name}_scatter.png")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    # Load datasets
    dataset_B = load_cif_dataset(args.b_path)
    dataset_C = load_cif_dataset(args.c_path)
    dataset_N = load_cif_dataset(args.n_path)

    # Split 80/20 per element
    train_B, test_B = element_split(dataset_B, 0.8)
    train_C, test_C = element_split(dataset_C, 0.8)
    train_N, test_N = element_split(dataset_N, 0.8)

    # Merge training sets
    train_dataset = ConcatDataset([train_B, train_C, train_N])
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_pool)

    # Load pretrained model
    checkpoint = torch.load(args.modelpath,
                            map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**checkpoint['args'])
    structures, _, _ = dataset_C[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=model_args.atom_fea_len,
        n_conv=model_args.n_conv,
        h_fea_len=model_args.h_fea_len,
        n_h=model_args.n_h,
        classification=False
    )
    model.load_state_dict(checkpoint['state_dict'])
    if args.cuda:
        model.cuda()

    # Normalizer
    normalizer = Normalizer()
    normalizer.load_state_dict(checkpoint['normalizer'])

    # Fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("\nStarting Fine-Tuning...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for input, target, _ in train_loader:
            if args.cuda:
                input_var = (input[0].cuda(),
                             input[1].cuda(),
                             input[2].cuda(),
                             [idx.cuda() for idx in input[3]])
                target = target.cuda()
            else:
                input_var = input

            target_normed = normalizer.norm(target)
            output = model(*input_var)
            loss = criterion(output, target_normed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(train_loader):.6f}")

    # --------------------------------------------------
    # Evaluate each element separately
    # --------------------------------------------------
    evaluate(model, normalizer, test_B, "Boron")
    evaluate(model, normalizer, test_C, "Carbon")
    evaluate(model, normalizer, test_N, "Nitrogen")


if __name__ == "__main__":
    main()