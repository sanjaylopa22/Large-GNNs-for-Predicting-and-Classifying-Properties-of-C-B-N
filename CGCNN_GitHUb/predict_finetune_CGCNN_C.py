import argparse
import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
from pymatgen.core import Structure

from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet


# --------------------------------------------------
# Argument Parser
# --------------------------------------------------
parser = argparse.ArgumentParser(description='CGCNN Carbon Fine-Tuning')
parser.add_argument('modelpath', help='Path to pretrained model')
parser.add_argument('cifpath', help='Path to CIF folder')
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
# MAE
# --------------------------------------------------
def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))


# --------------------------------------------------
# Main
def main():

    print("Loading dataset (Carbon materials, no filtering)...")
    dataset = CIFData(args.cifpath)

    print(f"Total materials found: {len(dataset)}")

    if len(dataset) < 5:
        print("Not enough samples for splitting.")
        return

    # --------------------------------------------------
    # 80/20 Split (ALL materials)
    # --------------------------------------------------
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_pool)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate_pool)

    # --------------------------------------------------
    # Load Pretrained Model
    # --------------------------------------------------
    print("Loading pretrained model...")

    checkpoint = torch.load(args.modelpath,
                            map_location=lambda storage, loc: storage)

    model_args = argparse.Namespace(**checkpoint['args'])

    structures, _, _ = dataset[0]
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

    print("Pretrained model loaded successfully.")

    # --------------------------------------------------
    # Normalizer
    # --------------------------------------------------
    normalizer = Normalizer()
    normalizer.load_state_dict(checkpoint['normalizer'])

    # --------------------------------------------------
    # Fine-Tuning Setup
    # --------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # --------------------------------------------------
    # Fine-Tuning
    # --------------------------------------------------
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

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {total_loss/len(train_loader):.6f}")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    print("\nEvaluating on 20% test set (Carbon materials)...")

    model.eval()

    test_targets = []
    test_preds = []
    test_ids = []

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

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    mae_value = np.mean(np.abs(test_targets_arr - test_preds_arr))
    mse_value = np.mean((test_targets_arr - test_preds_arr) ** 2)

    print("\n======================================")
    print("FINAL TEST RESULTS (Carbon MATERIALS)")
    print("======================================")
    print(f"MAE : {mae_value:.6f} eV/atom")
    print(f"MSE : {mse_value:.6f} (eV/atom)^2")
    print("======================================\n")

    # --------------------------------------------------
    # Print All Predictions
    # --------------------------------------------------
    print("Material-wise Predictions:\n")

    for cif_id, target, pred in zip(test_ids,
                                    test_targets_arr,
                                    test_preds_arr):

        formula = os.path.splitext(os.path.basename(cif_id))[0]

        print(f"{formula:20s} | "
              f"Ground Truth: {target:.6f} | "
              f"Predicted: {pred:.6f}")

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------
    with open('test_results_Carbon.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Material', 'Target_FE', 'Predicted_FE'])

        for cif_id, target, pred in zip(test_ids,
                                        test_targets_arr,
                                        test_preds_arr):

            formula = os.path.splitext(os.path.basename(cif_id))[0]
            writer.writerow((formula, target, pred))

    # --------------------------------------------------
    # Scatter Plot
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(test_targets_arr, test_preds_arr)
    plt.plot([min(test_targets_arr), max(test_targets_arr)],
             [min(test_targets_arr), max(test_targets_arr)], 'r--')
    plt.xlabel("Ground Truth Formation Energy (eV/atom)")
    plt.ylabel("Predicted Formation Energy (eV/atom)")
    plt.title("CGCNN Fine-Tuned Model (Carbon Materials)")
    plt.grid(True)
    plt.savefig("carbon_materials_scatter.png", dpi=300)

    print("\nSaved:")
    print(" - test_results_Carbon.csv")
    print(" - CGCNN_Carbon_Finetune_Scatter.png")

if __name__ == "__main__":
    main()