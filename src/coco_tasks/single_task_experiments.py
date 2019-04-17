import os
from typing import List, Union, Tuple

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from coco_tasks.settings import TB_ROOT
from coco_tasks.single_task_datasets import CocoTasksTest, CocoTasksGT, CocoTasksTestGT
from coco_tasks.single_task_networks import ClassifierBaselineNetwork


class OverfitSampler(Sampler):
    def __init__(self, main_source, indices):
        super().__init__(main_source)
        self.main_source = main_source
        self.indices = indices

        main_source_len = len(self.main_source)

        how_many = int(round(main_source_len / len(self.indices)))
        self.to_iter_from = []
        for _ in range(how_many):
            self.to_iter_from.extend(self.indices)

    def __iter__(self):
        return iter(self.to_iter_from)

    def __len__(self):
        return len(self.main_source)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class ClassifierExperiment(object):
    def __init__(
        self,
        network: ClassifierBaselineNetwork,
        dataset: CocoTasksGT = None,
        lr: float = 1e-5,
        wd: float = 0,
        optimizer: Optimizer = None,
        tensorboard: bool = True,
    ):
        self.network = network
        self.dataset = dataset
        self.lr = lr
        self.wd = wd
        self.optimizer = optimizer
        self.tensorboard = tensorboard
        self.device = get_device()

        if self.optimizer is None:
            self.optimizer = SGD(network.parameters(), lr=lr, weight_decay=wd)

        if tensorboard:
            self.summary_writer = SummaryWriter(
                log_dir=os.path.join(TB_ROOT, self.get_name())
            )

    def train_n_epochs(self, n: int, lr_scheduler: bool = False, overfit: bool = True):
        if self.dataset is None:
            raise Exception("Training is not possible, Dataset is None.")

        if lr_scheduler:
            lrsch = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.train()
        self.network.extractor.extractor.train(mode=False)

        if overfit:
            sampler = OverfitSampler(main_source=self.dataset, indices=[42])

        batch_number = 0
        for epoch_number in range(n):
            if overfit:
                # noinspection PyUnboundLocalVariable
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
                # noinspection PyUnboundLocalVariable
                lrsch.step()

            for batch in tqdm(train_data_loader):
                x, d, t, detections = batch
                x, d, t = (
                    x.to(self.device),
                    d.to(self.device),
                    t.type(torch.LongTensor).to(self.device),
                )

                self.optimizer.zero_grad()
                preds = self.network.forward(x)
                loss = self.network.compute_loss(preds, t)

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
                    self.summary_writer.add_scalar(
                        "train_loss", loss.data, global_step=batch_number
                    )

                batch_number += 1

    def get_name(self) -> str:
        string_format = "single-task-classifier-n:{nname}-d:{dname}-o:{oname}-lr:{lr:f}-wd:{wd}".format(
            nname=self.network.__class__.__name__,
            dname=self.dataset.__repr__(),
            oname=self.optimizer.__class__.__name__,
            lr=self.optimizer.defaults["lr"],
            wd=self.optimizer.defaults["weight_decay"],
        )
        return string_format

    def get_mean_class_prob(self) -> Tuple[List[float], List[float]]:
        self.network.eval()
        self.network.to(self.device)

        train_data_loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=self.dataset.custom_collate,
        )

        fc_probs = []
        non_fc_probs = []

        for batch in tqdm(train_data_loader):
            x, d, t, detections = batch
            x = x.to(self.device)

            preds = self.network.estimate_probability(x)
            preds = preds.detach().cpu().numpy()

            for i in range(x.size(0)):
                if t[i] == 0:
                    non_fc_probs.append(preds[i][0])
                else:
                    fc_probs.append(preds[i][1])

        return non_fc_probs, fc_probs

    def do_test(
        self, test_dataset: Union[CocoTasksTest, CocoTasksTestGT]
    ) -> List[dict]:
        self.network.eval()

        if torch.cuda.is_available():
            self.network.cuda()

        results = []
        data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=test_dataset.custom_collate,
        )
        with torch.no_grad():
            for batch in tqdm(data_loader):
                x, d, detections = batch
                x = x.to(self.device)

                probabilities = self.network.estimate_probability(x)
                probabilities = probabilities.to(torch.device("cpu")).detach().numpy()

                for i in range(len(detections)):
                    score = float(probabilities[i][1] * detections[i]["score"])
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
