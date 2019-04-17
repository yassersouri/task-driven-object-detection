import os
from typing import List

import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from coco_tasks.ranker_datasets import CocoTasksRanker, CocoTasksRankerTestGT
from coco_tasks.ranker_networks import Ranker
from coco_tasks.settings import TB_ROOT
from coco_tasks.single_task_experiments import get_device, OverfitSampler

__all__ = ["Experiment"]


class Experiment(object):
    def __init__(
        self,
        network: Ranker,
        dataset: CocoTasksRanker = None,
        lr: float = 1e-4,
        wd: float = 0,
        optimizer: Optimizer = None,
        tensorboard: bool = False,
        batch_size: int = 32,
    ):
        self.network = network
        self.dataset = dataset
        self.lr = lr
        self.wd = wd
        self.optimizer = optimizer
        self.tensorboard = tensorboard
        self.batch_size = batch_size
        self.device = get_device()

        if self.optimizer is None:
            self.optimizer = Adam(
                self.network.parameters(), lr=self.lr, weight_decay=self.wd
            )

        if tensorboard:
            self.summary_writer = SummaryWriter(
                log_dir=os.path.join(TB_ROOT, self.get_name())
            )

    def get_name(self) -> str:
        string_format = "ranker-n:{nname}-d:{dname}-o:{oname}-lr:{lr:f}-wd:{wd}".format(
            nname=self.network.__class__.__name__,
            dname=self.dataset.__repr__(),
            oname=self.optimizer.__class__.__name__,
            lr=self.optimizer.defaults["lr"],
            wd=self.optimizer.defaults["weight_decay"],
        )
        return string_format

    def train_n_epochs(self, n: int, lr_scheduler: bool = False, overfit: bool = False):
        if self.dataset is None:
            raise Exception("Training is not possible, Dataset is None.")

        if lr_scheduler:
            lrsch = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

        self.network.to(self.device)
        self.network.train()
        self.network.extractor_net.extractor.train(False)

        if overfit:
            sampler = OverfitSampler(
                main_source=self.dataset, indices=list(range(self.batch_size))
            )

        batch_number = 0
        for epoch in range(n):
            if overfit:
                # noinspection PyUnboundLocalVariable
                train_data_loader = DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=1,
                    collate_fn=self.dataset.custom_collate,
                    sampler=sampler,
                )
            else:
                train_data_loader = DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=1,
                    collate_fn=self.dataset.custom_collate,
                )

            if lr_scheduler:
                # noinspection PyUnboundLocalVariable
                lrsch.step()

            for batch in tqdm(train_data_loader):
                x1, x2, r = batch
                x1, x2, r = x1.to(self.device), x2.to(self.device), r.to(self.device)

                self.optimizer.zero_grad()
                pred_r = self.network.forward(x1, x2)
                loss = self.network.compute_loss(pred_r, r)

                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                loss.backward()
                self.optimizer.step()

                if self.tensorboard:
                    self.summary_writer.add_scalar(
                        "train_loss", loss.item(), global_step=batch_number
                    )

                batch_number += 1

    def do_test(self, test_dataset: CocoTasksRankerTestGT) -> List[dict]:
        self.network.to(self.device)
        self.network.eval()

        results = []
        data_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=True,
            num_workers=1,
            collate_fn=test_dataset.custom_collate,
        )
        with torch.no_grad():
            for batch in tqdm(data_loader):
                x, detections = batch
                x = x.to(self.device)

                ranks = self.network.rank_estimate(x).cpu().detach().view(-1).numpy()

                for i in range(len(detections)):
                    score = float(ranks[i])

                    results.append(
                        {
                            "image_id": detections[i]["image_id"],
                            "category_id": 1,
                            "score": detections[i]["score"],
                            "rank_value": score,
                            "bbox": detections[i]["bbox"],
                            "COCO_category_id": detections[i]["category_id"],
                        }
                    )

        return results
