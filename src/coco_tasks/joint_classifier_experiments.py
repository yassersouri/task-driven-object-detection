import os
from typing import Optional, Union

import torch
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from coco_tasks.graph_datasets import JointCocoTasks as GJointCocoTasks
from coco_tasks.joint_classifier_datasets import JointCocoTasks, task_number_to_task_id
from coco_tasks.joint_classifier_networks import (
    JointClassifier,
    JointClassifierWithClass,
)
from coco_tasks.settings import TB_ROOT
from coco_tasks.single_task_datasets import CocoTasksTest, CocoTasksTestGT
from coco_tasks.single_task_experiments import get_device, OverfitSampler


class JointClassifierExperiment(object):
    def __init__(
        self,
        network: JointClassifier,
        optimizer: Optimizer,
        dataset: Optional[JointCocoTasks] = None,
        tensorboard: bool = True,
        seed: Optional[int] = None,
    ):
        self.network = network
        self.optimizer = optimizer
        self.dataset = dataset
        self.tensorboard = tensorboard
        self.seed = seed
        self.device = get_device()

        self.writer = None
        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(TB_ROOT, self.get_name()))

    def get_name(self) -> str:
        string_format = "joint-classifier-n:{nname}(uw:{uw})-d:{dname}-o:{oname}(lr:{lr:f}-wd:{wd:f})".format(
            nname=self.network.__class__.__name__,
            uw=self.network.use_weight,
            dname=self.dataset.__repr__(),
            oname=self.optimizer.__class__.__name__,
            lr=self.optimizer.defaults["lr"],
            wd=self.optimizer.defaults["weight_decay"],
        )

        if self.seed is not None:
            string_format += "-s:{}".format(self.seed)
        return string_format

    def train_n_epochs(self, n: int, overfit: bool = False, lr_scheduler: bool = False):
        if self.dataset is None:
            raise Exception("Training is not possible, Dataset is None.")
        self.network.train()
        self.network.extractor.extractor.train(mode=False)
        self.network.to(self.device)

        sampler = None
        if overfit:
            sampler = OverfitSampler(main_source=self.dataset, indices=[42])

        lrsch = None
        if lr_scheduler:
            lrsch = StepLR(self.optimizer, step_size=1, gamma=0.1)

        iter_num = 0
        for epoch_number in range(n):
            if overfit:
                train_data_loader = DataLoader(
                    self.dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    collate_fn=self.dataset.custom_collate,
                    sampler=sampler,
                )
            else:
                train_data_loader = DataLoader(
                    self.dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=1,
                    collate_fn=self.dataset.custom_collate,
                )

            if lr_scheduler:
                lrsch.step()

            for batch in tqdm(train_data_loader):
                x, t, m = batch  # type: Tensor
                x, t, m = x.to(self.device), t.to(self.device), m.to(self.device)

                self.optimizer.zero_grad()
                logits = self.network.forward(x)
                loss = self.network.compute_loss(logits, t, m)

                loss.backward()
                parameters = list(
                    filter(lambda p: p.grad is not None, self.network.parameters())
                )

                total_norm = 0
                for p in parameters:
                    param_norm = p.grad.data.norm(2.0)
                    total_norm += param_norm.item() ** 2.0
                total_norm **= 1.0 / 2.0

                if total_norm > 15:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 15.0)
                self.optimizer.step()

                if self.tensorboard:
                    self.writer.add_scalar(
                        "train_loss", loss.item(), global_step=iter_num
                    )

                iter_num += 1

    def do_test(self, test_db: Union[CocoTasksTest, CocoTasksTestGT], task_number: int):
        self.network.eval()
        self.network.to(self.device)

        task_id = task_number_to_task_id(task_number)

        results = []
        data_loader = DataLoader(
            test_db, batch_size=1, num_workers=1, collate_fn=test_db.custom_collate
        )
        with torch.no_grad():
            for batch in tqdm(data_loader):
                x, d, detections = batch
                x = x.to(self.device)

                probabilities = self.network.estimate_probability(x)
                task_probabilities = probabilities[:, task_id].detach().cpu().numpy()

                for i in range(len(detections)):
                    score = float(task_probabilities[i] * detections[i]["score"])

                    results.append(
                        {
                            "image_id": detections[i]["image_id"],
                            "category_id": 1,
                            "score": score,
                            "bbox": detections[i]["bbox"],
                            "COCO_category_id": detections[i]["category_id"],
                        }
                    )

        return results


class JointClassifierExperimentWithClass(object):
    def __init__(
        self,
        network: JointClassifierWithClass,
        optimizer: Optimizer,
        dataset: Optional[GJointCocoTasks] = None,
        tensorboard: bool = True,
        seed: Optional[int] = None,
    ):
        self.network = network
        self.optimizer = optimizer
        self.dataset = dataset
        self.tensorboard = tensorboard
        self.seed = seed
        self.device = get_device()

        self.writer = None
        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(TB_ROOT, self.get_name()))

    def get_name(self) -> str:
        string_format = "joint-classifier-withclass-n:{nname}-d:{dname}-o:{oname}(lr:{lr:f}-wd:{wd:f})".format(
            nname=self.network.__class__.__name__,
            dname=self.dataset.__repr__(),
            oname=self.optimizer.__class__.__name__,
            lr=self.optimizer.defaults["lr"],
            wd=self.optimizer.defaults["weight_decay"],
        )

        if self.seed is not None:
            string_format += "-s:{}".format(self.seed)
        return string_format

    def train_n_epochs(self, n: int, overfit: bool = False, lr_scheduler: bool = False):
        if self.dataset is None:
            raise Exception("Training is not possible, Dataset is None.")
        self.network.train()
        self.network.extractor.extractor.train(mode=False)
        self.network.to(self.device)

        sampler = None
        if overfit:
            sampler = OverfitSampler(main_source=self.dataset, indices=[42])

        lrsch = None
        if lr_scheduler:
            lrsch = StepLR(self.optimizer, step_size=1, gamma=0.1)

        iter_num = 0
        for epoch_number in range(n):
            if overfit:
                train_data_loader = DataLoader(
                    self.dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    collate_fn=self.dataset.custom_collate,
                    sampler=sampler,
                )
            else:
                train_data_loader = DataLoader(
                    self.dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=1,
                    collate_fn=self.dataset.custom_collate,
                )

            if lr_scheduler:
                lrsch.step()

            for batch in tqdm(train_data_loader):
                x, t, m, c, d = batch  # type: Tensor
                x, t, m, c = (
                    x.to(self.device),
                    t.to(self.device),
                    m.to(self.device),
                    c.to(self.device),
                )

                self.optimizer.zero_grad()
                logits = self.network.forward(x, c)
                loss = self.network.compute_loss(logits, t, m)

                loss.backward()
                parameters = list(
                    filter(lambda p: p.grad is not None, self.network.parameters())
                )

                total_norm = 0
                for p in parameters:
                    param_norm = p.grad.data.norm(2.0)
                    total_norm += param_norm.item() ** 2.0
                total_norm **= 1.0 / 2.0

                if total_norm > 15:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 15.0)
                self.optimizer.step()

                if self.tensorboard:
                    self.writer.add_scalar(
                        "train_loss", loss.item(), global_step=iter_num
                    )

                iter_num += 1

    def do_test(self, test_db: Union[CocoTasksTest, CocoTasksTestGT], task_number: int):
        self.network.eval()
        self.network.to(self.device)

        task_id = task_number_to_task_id(task_number)

        results = []
        data_loader = DataLoader(
            test_db, batch_size=1, num_workers=1, collate_fn=test_db.custom_collate
        )
        with torch.no_grad():
            for batch in tqdm(data_loader):
                x, c, d, detections = batch
                x, c, d = x.to(self.device), c.to(self.device), d.to(self.device)

                probabilities = self.network.estimate_probability(x, c)
                task_probabilities = probabilities[:, task_id].detach().cpu().numpy()

                for i in range(len(detections)):
                    score = float(task_probabilities[i] * detections[i]["score"])

                    results.append(
                        {
                            "image_id": detections[i]["image_id"],
                            "category_id": 1,
                            "score": score,
                            "bbox": detections[i]["bbox"],
                            "COCO_category_id": detections[i]["category_id"],
                        }
                    )

        return results
