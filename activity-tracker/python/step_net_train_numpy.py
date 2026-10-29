"""A from-scratch (no PyTorch) implementation of StepNet's training, so the
board itself can retrain on demand from the phone page.

PyTorch is too large to install on this board (it pulled in a multi-GB CUDA
stack meant for an NVIDIA GPU, which this ARM board doesn't have), so
train_step_net.py (real PyTorch, run on a computer) stays the main training
pipeline for the writeup. This file is a second, independent implementation
of the exact same math PyTorch does automatically with .backward() - every
gradient here is written out by hand using the chain rule, which is also a
good way to show you understand what backprop is actually computing.

Same network shape as step_net.py: 4 inputs -> 8 -> 4 -> 1, ReLU/ReLU/Sigmoid.
"""
import numpy as np

from steps import STEP_FEATURE_KEYS, STEP_NET_INPUT_DIM as INPUT_DIM, STEP_NET_HIDDEN1 as HIDDEN1, STEP_NET_HIDDEN2 as HIDDEN2


def _relu(z):
    return np.maximum(z, 0.0)


def _relu_grad(z):
    return (z > 0).astype(float)


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def train(X, y, epochs=400, lr=0.1, seed=0):
    """X: (N, 4) normalized features. y: (N, 1) of 0/1 labels.
    Returns a dict in the same shape train_step_net.py exports, so
    step_net_inference.py can load whichever one produced it."""
    rng = np.random.default_rng(seed)

    # Small random starting weights - starting all-zero would make every
    # neuron in a layer compute the identical gradient forever (nothing to
    # break the symmetry between them).
    def init(out_dim, in_dim):
        return rng.normal(0, np.sqrt(2.0 / in_dim), size=(out_dim, in_dim))

    W1, b1 = init(HIDDEN1, INPUT_DIM), np.zeros(HIDDEN1)
    W2, b2 = init(HIDDEN2, HIDDEN1), np.zeros(HIDDEN2)
    W3, b3 = init(1, HIDDEN2), np.zeros(1)

    N = X.shape[0]
    for _ in range(epochs):
        # ---- forward pass ----
        z1 = X @ W1.T + b1
        a1 = _relu(z1)
        z2 = a1 @ W2.T + b2
        a2 = _relu(z2)
        z3 = a2 @ W3.T + b3
        y_hat = _sigmoid(z3)

        # ---- backward pass (chain rule, layer by layer, back to front) ----
        # d(BCE loss)/d(z3) simplifies to exactly (y_hat - y) when the last
        # layer is a sigmoid - one of the reasons sigmoid+BCE are usually
        # paired together.
        dz3 = (y_hat - y) / N
        dW3, db3 = dz3.T @ a2, dz3.sum(axis=0)

        da2 = dz3 @ W3
        dz2 = da2 * _relu_grad(z2)
        dW2, db2 = dz2.T @ a1, dz2.sum(axis=0)

        da1 = dz2 @ W2
        dz1 = da1 * _relu_grad(z1)
        dW1, db1 = dz1.T @ X, dz1.sum(axis=0)

        # ---- gradient descent: nudge every weight a little against its gradient ----
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
        W3 -= lr * dW3; b3 -= lr * db3

    return {
        "W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3,
    }


def accuracy(weights, X, y):
    z1 = X @ weights["W1"].T + weights["b1"]
    a1 = _relu(z1)
    z2 = a1 @ weights["W2"].T + weights["b2"]
    a2 = _relu(z2)
    z3 = a2 @ weights["W3"].T + weights["b3"]
    y_hat = _sigmoid(z3)
    pred = (y_hat >= 0.5).astype(float)
    return float((pred == y).mean())


def to_export_payload(weights, mean, std):
    return {
        "feature_keys": STEP_FEATURE_KEYS,
        "mean": mean.tolist(), "std": std.tolist(),
        "fc1_w": weights["W1"].tolist(), "fc1_b": weights["b1"].tolist(),
        "fc2_w": weights["W2"].tolist(), "fc2_b": weights["b2"].tolist(),
        "fc3_w": weights["W3"].tolist(), "fc3_b": weights["b3"].tolist(),
    }
