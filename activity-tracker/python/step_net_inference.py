"""Runs StepNet's forward pass on the board, using only numpy.

PyTorch did the learning (see train_step_net.py) - it ran gradient descent
to find the fc1/fc2/fc3 weight matrices that best separate real steps from
false positives. But "using" a trained net to make one prediction is just
a few matrix multiplies and two simple functions (ReLU, sigmoid); it
doesn't need PyTorch's training machinery at all. This file reloads the
exported numbers from step_net_weights.json and does that handful of
multiplies by hand, so the board doesn't need PyTorch installed and every
line of what runs live is something we can point at and explain.
"""
import json

import numpy as np

from steps import STEP_FEATURE_KEYS


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    # The plain 1/(1+exp(-x)) formula overflows for very negative x (exp of a
    # large positive number) - this splits into two algebraically equivalent
    # forms so neither branch ever computes exp() of a large positive number.
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class StepNetInference:
    def __init__(self, weights_path):
        with open(weights_path) as f:
            w = json.load(f)
        assert w["feature_keys"] == STEP_FEATURE_KEYS, "step_net_weights.json doesn't match STEP_FEATURE_KEYS"

        self.mean = np.array(w["mean"])
        self.std = np.array(w["std"])
        self.fc1_w, self.fc1_b = np.array(w["fc1_w"]), np.array(w["fc1_b"])
        self.fc2_w, self.fc2_b = np.array(w["fc2_w"]), np.array(w["fc2_b"])
        self.fc3_w, self.fc3_b = np.array(w["fc3_w"]), np.array(w["fc3_b"])

    def predict_proba(self, features: dict) -> float:
        """features: a dict with the 4 keys in STEP_FEATURE_KEYS (see
        steps.py). Returns the net's estimated probability (0-1) that this
        candidate peak was a genuine step.
        """
        x = np.array([features[k] for k in STEP_FEATURE_KEYS])
        x = (x - self.mean) / self.std  # same normalization used in training

        # PyTorch's nn.Linear stores weights as (out_features, in_features),
        # so y = W @ x + b matches what StepNet.forward() computes.
        h1 = _relu(self.fc1_w @ x + self.fc1_b)
        h2 = _relu(self.fc2_w @ h1 + self.fc2_b)
        out = _sigmoid(self.fc3_w @ h2 + self.fc3_b)
        return float(out[0])

    def is_step(self, features: dict, threshold: float = 0.5) -> bool:
        return self.predict_proba(features) >= threshold
