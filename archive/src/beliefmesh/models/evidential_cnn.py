"""Evidential CNN. Spec Sec 3: three conv blocks + two FC layers, directly emitting
raw NIG parameters via fc2. All parameters trainable throughout every stage --
no frozen/trainable split for this experiment suite (backbone-freezing and
adapter-based fine-tuning are deployment-chapter concerns, out of scope here).
Ported from the prior experiment's EvidentialCNN unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialCNN(nn.Module):
    """Three convolutional blocks + two fully connected layers, fc2 emitting the four
    raw values that nig_reparameterise (see evidential.py) turns into (gamma, nu, alpha, beta).

    in_channels: 3 (RGB) by default, unchanged for every existing caller.
    MeshNode uses in_channels=5 -- 3 RGB + 2 CoordConv-style channels
    encoding the cell's position WITHIN the node's own FOV (local, not
    global -- see mesh.py's local_coord_channels). Cheap, well-established
    fix (Liu et al. 2018 CoordConv) for the "no coordinate awareness"
    limitation confirmed empirically via the multi-region diagnostic: a
    node whose FOV spans two genuinely different true regions cannot
    represent both, since two images from different cells were otherwise
    indistinguishable to the model."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        #conv layers
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        #fully connected layers
        self.fc1 = nn.Linear(1152, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #conv forward
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        #flatten
        x = x.view(x.size(0), -1) # flatten to (batch_size, 1152)

        #full connected forward
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #activations and return NIG params (gamma, nu, alpha, beta)
        gamma = x[:, 0]
        nu = F.softplus(x[:, 1])
        alpha = F.softplus(x[:, 2]) + 1
        beta = F.softplus(x[:, 3])

        return gamma, nu, alpha, beta
