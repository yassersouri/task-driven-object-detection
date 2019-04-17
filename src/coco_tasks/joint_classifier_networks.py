import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from coco_tasks.graph_networks import ExtractorResNet
from coco_tasks.settings import TASK_NUMBERS


class JointClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tasks = len(TASK_NUMBERS)
        self.use_weight = False
        self.extractor = ExtractorResNet()
        self.attribute_layer = nn.Linear(
            in_features=self.extractor.out_channels, out_features=self.num_tasks
        )
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.drop = nn.Dropout(0.25)

    def extractor_forward(self, x: Tensor) -> Tensor:
        """
        :param x: [B x 3 x 224 x 224]
        :return: [B x feat_dim]
        """
        feats = self.extractor.forward(x)
        feats = self.drop(feats)
        return feats

    def estimate_probability(self, x: Tensor) -> Tensor:
        """
        This should be used at test time. Don't use during training.
        :param x: [B x 3 x 224 x 224]
        :return: [B x num_tasks], each item is in [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def forward(self, x: Tensor) -> Tensor:
        """
        This is the forward pass of the network used during training.
        Given a batch of images (or crops of images) returns the logits for all of the tasks.
        :param x: [B x 3 x 224 x 224]
        :return: the logits [B x num_tasks]
        """

        feats = self.extractor_forward(x)  # [B x feat_dim]
        logits = self.attribute_layer.forward(feats)  # [B x num_tasks]
        return logits

    def compute_loss(self, logits: Tensor, t: Tensor, m: Tensor) -> Tensor:
        """
        Computes the loss for all of the tasks.
        :param logits: [B x num_tasks]
        :param t: [B x num_tasks]
        :param m: [num_tasks]
        :return: [] loss.
        """
        loss = logits.new_zeros(())
        for tn in range(self.num_tasks):
            if m[tn]:
                loss += self._compute_single_loss(tn, logits[:, tn], t[:, tn])

        return loss

    def _compute_single_loss(
        self, task_index: int, logits: Tensor, t: Tensor
    ) -> Tensor:
        """
        Computes the loss for a single task.
        :param logits: [B]
        :param t: [B]
        :return: [] loss.
        """
        return self.loss.forward(logits, t)


class JointClassifierWithClass(nn.Module):
    def __init__(self, h_dim: int = 128, c_dim: int = 90):
        super().__init__()
        self.h_dim = h_dim
        self.c_dim = c_dim
        self.num_tasks = len(TASK_NUMBERS)
        self.extractor = ExtractorResNet()
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.drop = nn.Dropout(0.25)
        self.phi_layer = nn.Linear(
            in_features=self.extractor.out_channels, out_features=h_dim, bias=False
        )
        self.c_layer = nn.Linear(in_features=self.c_dim, out_features=h_dim, bias=False)
        self.non_lin = F.relu
        self.fc = nn.Linear(in_features=self.h_dim, out_features=self.num_tasks)

    def extractor_forward(self, x: Tensor) -> Tensor:
        """
        :param x: [B x 3 x 224 x 224]
        :return: [B x feat_dim]
        """
        feats = self.extractor.forward(x)
        return feats

    def estimate_probability(self, x: Tensor, c: Tensor) -> Tensor:
        """
        This should be used at test time. Don't use during training.
        :param x: [B x 3 x 224 x 224]
        :param c: the one hot encoding of the classes: [B x num_classes]
        :return: [B x num_tasks], each item is in [0, 1]
        """
        logits = self.forward(x, c)
        return torch.sigmoid(logits)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        This is the forward pass of the network used during training.
        Given a batch of images (or crops of images) returns the logits for all of the tasks.
        :param x: [B x 3 x 224 x 224]
        :param c: the one hot encoding of the classes: [B x num_classes]
        :return: the logits [B x num_tasks]
        """

        feats = self.extractor_forward(x)  # [B x feat_dim]
        x = self.non_lin(self.phi_layer(feats)) * self.non_lin(
            self.c_layer(c)
        )  # [B x h_dim]
        x = self.drop(x)
        logits = self.fc.forward(x)  # dim: [B x num_tasks]
        return logits

    def compute_loss(self, logits: Tensor, t: Tensor, m: Tensor) -> Tensor:
        """
        Computes the loss for all of the tasks.
        :param logits: [B x num_tasks]
        :param t: [B x num_tasks]
        :param m: [num_tasks]
        :return: [] loss.
        """
        loss = logits.new_zeros(())
        for tn in range(self.num_tasks):
            if m[tn]:
                loss += self._compute_single_loss(tn, logits[:, tn], t[:, tn])

        return loss

    def _compute_single_loss(
        self, task_index: int, logits: Tensor, t: Tensor
    ) -> Tensor:
        """
        Computes the loss for a single task.
        :param logits: [B]
        :param t: [B]
        :return: [] loss.
        """
        return self.loss.forward(logits, t)
