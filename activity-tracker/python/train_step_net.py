"""Trains StepNet on step_candidates.csv using PyTorch, prints a k-fold
cross-validation accuracy (an honest "how good is this, really" estimate,
same spirit as Model2's leave-one-out check), then trains a final model on
ALL the data and exports its weights for the board to use at inference
time (see step_net_inference.py).

Run on a computer, after build_step_dataset.py:  python3 train_step_net.py
"""
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn

from steps import STEP_FEATURE_KEYS
from step_net import StepNet

DATA_PATH = Path(__file__).resolve().parent / "data" / "step_candidates.csv"
WEIGHTS_PATH = Path(__file__).resolve().parent / "data" / "step_net_weights.json"

K_FOLDS = 5
EPOCHS = 300
LEARNING_RATE = 0.02
SEED = 0


def load_dataset():
    with open(DATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    X = [[float(r[k]) for k in STEP_FEATURE_KEYS] for r in rows]
    y = [int(r["is_step"]) for r in rows]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)


def normalize(X, mean, std):
    return (X - mean) / std


def stratified_folds(y, k, seed):
    """Splits indices into k folds, keeping each class's proportion roughly
    equal in every fold - important here since the dataset is small and a
    plain random split could easily dump most of one class into one fold.
    """
    rng = random.Random(seed)
    by_class = {}
    for i, label in enumerate(y.squeeze(1).tolist()):
        by_class.setdefault(label, []).append(i)
    for idxs in by_class.values():
        rng.shuffle(idxs)

    folds = [[] for _ in range(k)]
    for idxs in by_class.values():
        for i, idx in enumerate(idxs):
            folds[i % k].append(idx)
    return folds


def train_one_model(X_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE):
    model = StepNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()       # backprop: computes d(loss)/d(every weight)
        optimizer.step()      # nudges every weight a little against its gradient
    return model


def accuracy(model, X, y):
    with torch.no_grad():
        pred = (model(X) >= 0.5).float()
        return (pred == y).float().mean().item()


def cross_validate(X, y):
    folds = stratified_folds(y, K_FOLDS, SEED)
    fold_accuracies = []
    for i in range(K_FOLDS):
        val_idx = folds[i]
        train_idx = [j for f in range(K_FOLDS) if f != i for j in folds[f]]
        if not val_idx or not train_idx:
            continue

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        mean, std = X_train.mean(dim=0), X_train.std(dim=0)
        std[std == 0] = 1.0
        X_train_n, X_val_n = normalize(X_train, mean, std), normalize(X_val, mean, std)

        model = train_one_model(X_train_n, y_train)
        acc = accuracy(model, X_val_n, y_val)
        fold_accuracies.append(acc)
        print(f"  fold {i + 1}/{K_FOLDS}: {len(train_idx)} train, {len(val_idx)} val -> accuracy {acc:.2%}")
    return fold_accuracies


def export_weights(model, mean, std, path):
    sd = model.state_dict()
    payload = {
        "feature_keys": STEP_FEATURE_KEYS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fc1_w": sd["fc1.weight"].tolist(), "fc1_b": sd["fc1.bias"].tolist(),
        "fc2_w": sd["fc2.weight"].tolist(), "fc2_b": sd["fc2.bias"].tolist(),
        "fc3_w": sd["fc3.weight"].tolist(), "fc3_b": sd["fc3.bias"].tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    torch.manual_seed(SEED)
    X, y = load_dataset()
    n_pos = int(y.sum().item())
    print(f"Loaded {len(X)} candidate peaks ({n_pos} real steps, {len(X) - n_pos} non-steps)")

    print(f"\n{K_FOLDS}-fold cross-validation (honest accuracy estimate, same idea as leave-one-out):")
    fold_accuracies = cross_validate(X, y)
    if fold_accuracies:
        mean_acc = sum(fold_accuracies) / len(fold_accuracies)
        print(f"\nAverage cross-validation accuracy: {mean_acc:.2%}")

    print("\nTraining final model on all data for deployment...")
    mean, std = X.mean(dim=0), X.std(dim=0)
    std[std == 0] = 1.0
    final_model = train_one_model(normalize(X, mean, std), y)
    final_acc = accuracy(final_model, normalize(X, mean, std), y)
    print(f"Final model accuracy on its own training data: {final_acc:.2%} (expect this to look better than "
          f"cross-validation - it's seen all this data before)")

    export_weights(final_model, mean, std, WEIGHTS_PATH)
    print(f"\nExported weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
