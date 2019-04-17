# This file contains our re-implementation of Deep Relative Attributes (ACCV 2016) paper.

import torch
import torch.nn as nn
from torch import Tensor

from coco_tasks.graph_networks import ExtractorResNet

__all__ = ["Ranker", "BinaryCrossEntropy", "Clip"]


class Clip(nn.Module):
    def __init__(self, epsilon: float = 1e-7):
        super(Clip, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        thresh1 = nn.Threshold(self.epsilon, self.epsilon)
        x_lower_clipped = thresh1.forward(x)
        thresh2 = nn.Threshold(-1 + self.epsilon, -1 + self.epsilon)
        x_lower_clipped_minus = -1 * x_lower_clipped
        x_lower_and_upper_clipped_minus = thresh2.forward(x_lower_clipped_minus)
        return -1 * x_lower_and_upper_clipped_minus


class BinaryCrossEntropy(nn.Module):
    """
    Does Binary Cross Entropy in the Lasagne style. Also performs clipping of input predictions.
    """

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.clipper = Clip(self.epsilon)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        preds = self.clipper.forward(preds)

        part_1 = -1 * targets * torch.log(preds)
        part_2 = -1 * (1 - targets) * torch.log(1 - preds)
        loss = part_1 + part_2
        return torch.mean(loss)


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor_net = ExtractorResNet()
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(
            in_features=self.extractor_net.out_channels, out_features=1, bias=False
        )
        self.criterion = BinaryCrossEntropy()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1_rank_estimate = self.rank_estimate(x1)
        x2_rank_estimate = self.rank_estimate(x2)
        posterior = torch.sigmoid(x1_rank_estimate - x2_rank_estimate)

        return posterior

    def rank_estimate(self, x: Tensor) -> Tensor:
        x = self.extractor_net.forward(x)
        x = self.drop(x)
        return self.fc.forward(x)

    def compute_loss(self, pred_r: Tensor, r: Tensor) -> Tensor:
        return self.criterion.forward(pred_r, r)
