"""The step-vs-not-a-step neural net architecture.

This file only defines the network's shape; train_step_net.py (run on a
computer, uses PyTorch for training) and step_net_inference.py (runs live
on the board, pure numpy, no PyTorch needed there) both need to agree on
this shape, so it lives here once.

Architecture, in plain terms: 4 input numbers describing a candidate peak
-> 8 hidden neurons -> 4 hidden neurons -> 1 output number between 0 and 1
(the estimated probability this peak was a real step). Small on purpose:
with only ~90 labeled examples, a bigger net would just memorize the
training data instead of learning a general pattern.
"""
import torch.nn as nn

from steps import STEP_NET_INPUT_DIM as INPUT_DIM, STEP_NET_HIDDEN1 as HIDDEN1, STEP_NET_HIDDEN2 as HIDDEN2


class StepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))
