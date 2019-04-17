import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from coco_tasks.graph_networks import ExtractorResNet


class ClassifierBaselineNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = ExtractorResNet()
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(in_features=self.extractor.out_channels, out_features=2)
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def estimate_probability(self, x: Tensor) -> Tensor:
        """
        The forward pass of the attribute probability prediction
        :param x: input image patches, dim: [B x 3 x 224 x 224]
        :return: output estimates of the probability, dim: [B x 2]
        """
        x = self.estimate_scores(x)  # dim: [B x 2]
        x = F.softmax(x, dim=1)  # dim: [B x 2]
        return x

    def estimate_scores(self, x: Tensor) -> Tensor:
        """
        Computes the class scores, without the softmax
        :param x: input image patches, dim: [B x 3 x 224 x 224]
        :return: output class scores, dim: [B x 2]
        """
        x = self.extractor.forward(x)  # dim: [B x 2048]
        x = self.drop(x)
        x = self.fc.forward(x)  # dim: [B x 2]

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Estimate the fused probability of being first choice.
        :param x: input image patches, dim: [B x 3 x 224 x 224]
        :return:, dim: ([B x 2])
        """
        a = self.estimate_scores(x)  # dim: [B x 2]
        return a

    def compute_loss(self, a: Tensor, t: Tensor) -> Tensor:
        """
        Computes the loss.
        :param a: the probabilities, dim: [B x 1]
        :param t: the target, containing 1 is the item is first choice and 0 otherwise, dim: [B x 1]
        :return: the loss, the total loss is size averaged, dim: [1]
        """
        return self.loss(a, t.view(a.size(0)))
