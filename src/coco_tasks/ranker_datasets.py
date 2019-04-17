import json
import os
from contextlib import redirect_stdout
from pickle import load
from typing import Tuple, List

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

# noinspection PyUnresolvedReferences
from coco_tasks import _init_path
from coco_tasks.settings import (
    COCO_TASKS_PAIR_ROOT,
    COCO_TASKS_ANNOTATIONS_ROOT,
    COCO_TASKS_TEST_DETECTIONS,
)
from coco_tasks.single_task_datasets import (
    load_image,
    crop_img_to_bbox,
    get_image_file_name,
    image_transforms,
    target_transforms,
)
from pycocotools.coco import COCO

__all__ = ["CocoTasksRanker", "CocoTasksRankerTestGT", "CocoTasksRankerTest"]

PairType = Tuple[int, int, int, float]
devnull = open(os.devnull, "w")


class CocoTasksRanker(Dataset):
    def __init__(self, task_number: int):
        self.task_number = task_number

        self.pairs_file_path = os.path.join(
            COCO_TASKS_PAIR_ROOT, "pairs_train_{tn}.pkl".format(tn=self.task_number)
        )

        self.annotation_file = os.path.join(
            COCO_TASKS_ANNOTATIONS_ROOT,
            "task_{}_{}.json".format(self.task_number, "train"),
        )

        with open(self.pairs_file_path, "rb") as f:
            self.pairs = load(f)  # type: List[PairType]

        with redirect_stdout(devnull):
            self.task_coco = COCO(self.annotation_file)

    def __len__(self) -> int:
        return len(self.pairs)

    def __repr__(self) -> str:
        return "{dbname}({tname})".format(
            dbname=self.__class__.__name__, tname=self.task_number
        )

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :return: (img_crop1, img_crop2, rank_posterior)
                  img_cropi: [3 x 224 x 224]
                  rank_posterior: []
        """
        image_id, ann_id1, ann_id2, rankposterior = self.pairs[item]

        the_image_dict = self.task_coco.loadImgs(image_id)[0]
        the_ann1_dict = self.task_coco.loadAnns(ann_id1)[0]
        the_ann2_dict = self.task_coco.loadAnns(ann_id2)[0]

        # load image
        I = load_image(get_image_file_name(the_image_dict))

        # transform ann1 and ann2
        img1 = image_transforms(
            crop_img_to_bbox(I, target_transforms(the_ann1_dict["bbox"], I.size))
        )
        img2 = image_transforms(
            crop_img_to_bbox(I, target_transforms(the_ann2_dict["bbox"], I.size))
        )

        return img1, img2, torch.tensor(rankposterior)

    @staticmethod
    def custom_collate(
        batch: List[Tuple[Tensor, Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x0 = [x[0] for x in batch]
        x1 = [x[1] for x in batch]
        r = [x[2] for x in batch]

        return default_collate(x0), default_collate(x1), default_collate(r)


class CocoTasksRankerTestGT(Dataset):
    def __init__(self, task_number: int):
        self.task_number = task_number

        self.annotation_file = os.path.join(
            COCO_TASKS_ANNOTATIONS_ROOT,
            "task_{}_{}.json".format(self.task_number, "test"),
        )

        with redirect_stdout(devnull):
            self.task_coco = COCO(self.annotation_file)

        self.valid_annotations = []

        for ann in self.task_coco.loadAnns(self.task_coco.getAnnIds()):
            self.valid_annotations.append(ann)

    def __len__(self) -> int:
        return len(self.valid_annotations)

    def __getitem__(self, item: int) -> Tuple[Tensor, dict]:
        """
        :return: (img_crop, detection)
                  img_crop: [3 x 224 x 224]
                  detection: dict
        """
        ann = self.valid_annotations[item]

        the_image_dict = self.task_coco.loadImgs(
            self.task_coco.getImgIds(imgIds=ann["image_id"])
        )[0]
        img = load_image(get_image_file_name(the_image_dict))

        img_crop = image_transforms(
            crop_img_to_bbox(img, target_transforms(ann["bbox"], img.size))
        )

        detection = {
            "bbox": ann["bbox"],
            "score": 1.0,
            "category_id": ann["COCO_category_id"],
            "image_id": ann["image_id"],
        }

        return img_crop, detection

    @staticmethod
    def custom_collate(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, List[dict]]:
        imgs = [x[0] for x in batch]
        dicts = [x[1] for x in batch]

        return default_collate(imgs), dicts


class CocoTasksRankerTest(Dataset):
    THRESH = 0.02

    def __init__(self, task_number: int):
        self.task_number = task_number

        self.annotation_file = os.path.join(
            COCO_TASKS_ANNOTATIONS_ROOT,
            "task_{}_{}.json".format(self.task_number, "test"),
        )

        with redirect_stdout(devnull):
            self.task_coco = COCO(self.annotation_file)

        detections_file = COCO_TASKS_TEST_DETECTIONS
        with open(detections_file) as f:
            self.detections = json.load(f)

        task_images = self.task_coco.getImgIds()

        related_detections = []
        for d in self.detections:
            if d["image_id"] in task_images:
                related_detections.append(d)

        self.detections = related_detections

    def __len__(self) -> int:
        return len(self.detections)

    def __getitem__(self, item: int) -> Tuple[Tensor, dict]:
        """
        :return: (img_crop, detection)
                  img_crop: [3 x 224 x 224]
                  detection: dict
        """
        det = self.detections[item]

        the_image_dict = self.task_coco.loadImgs(
            self.task_coco.getImgIds(imgIds=det["image_id"])
        )[0]
        img = load_image(get_image_file_name(the_image_dict))

        img_crop = image_transforms(
            crop_img_to_bbox(img, target_transforms(det["bbox"], img.size))
        )

        detection = {
            "bbox": det["bbox"],
            "score": 1.0,
            "category_id": det["category_id"],
            "image_id": det["image_id"],
        }

        return img_crop, detection

    @staticmethod
    def custom_collate(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, List[dict]]:
        imgs = [x[0] for x in batch]
        dicts = [x[1] for x in batch]

        return default_collate(imgs), dicts
