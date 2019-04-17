import json
import os
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from coco_tasks.settings import (
    TASK_NUMBERS,
    COCO_TASKS_ANNOTATIONS_ROOT,
    COCO_TASKS_TEST_DETECTIONS,
)
from coco_tasks.single_task_datasets import (
    load_image,
    get_image_file_name,
    crop_img_to_bbox,
    image_transforms,
    target_transforms,
)
from pycocotools.coco import COCO

MAX_GPU_SIZE = 64
devnull = open(os.devnull, "w")


def get_one_hot(labels: List[int], num_classes: int = 90) -> np.ndarray:
    """
    returns: [len_labels x num_classes]
    """
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def get_bbox_array_from_annotations(
    anns: List[Dict], normalize_by: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    bbox = np.array([a["bbox"] for a in anns])
    if normalize_by is not None:
        I_h, I_w = normalize_by
        bbox[:, 0] /= I_w  # normalizing x
        bbox[:, 1] /= I_h  # normalizing y
        bbox[:, 2] /= I_w  # normalizing w
        bbox[:, 3] /= I_h  # normalizing h
    return bbox


def task_number_to_task_id(task_number: int) -> int:
    return TASK_NUMBERS.index(task_number)


class JointCocoTasks(Dataset):
    def __init__(self):
        self.task_image_ids = {}
        self.task_cocos = {}
        self.all_image_ids = []

        for task_number in TASK_NUMBERS:
            annotation_file = os.path.join(
                COCO_TASKS_ANNOTATIONS_ROOT,
                "task_{}_{}.json".format(task_number, "train"),
            )

            with redirect_stdout(devnull):
                self.task_cocos[task_number] = COCO(annotation_file)

            images = []
            for image_id in self.task_cocos[task_number].getImgIds():
                if len(self.task_cocos[task_number].getAnnIds(imgIds=image_id)) > 0:
                    images.append(image_id)

            self.task_image_ids[task_number] = images
            self.all_image_ids.extend(images)

        self.all_image_ids = list(set(self.all_image_ids))

        self.image_tasks = defaultdict(list)
        for image_id in self.all_image_ids:
            for task_number in TASK_NUMBERS:
                if image_id in self.task_image_ids[task_number]:
                    self.image_tasks[image_id].append(task_number)

    def __len__(self) -> int:
        return len(self.all_image_ids)

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __getitem__(
        self, item: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Gives the annotations of an image for all tasks. Also generates appropriate masks if the image
        doesn't have annotations for a particular image.
        :param item: the index of the image.
        :return: tuple(x, t, m, c, d),
            x is the transformed image data for different boxes, bim: [B x 3 x 224 x 224]
            t is the target. t[i, tn] is 1 if the image i i first choice for task tn. Otherwise it is 0. [B x task_numbers]
            m is the mask. m[tn] is 1 if this image is annotated for task tn. Otherwise it is 0. [task_numbers]
            c is the coco_class number one_hot encoded. [B x num_classes]
            d is the detection scores [B x 1]
            bbox is the normalized (by image size) bounding box (x, y, w, h) [B x 4]
        """

        # first choose MAX_GPU size number of annotations randomly.
        the_image_id = self.all_image_ids[item]
        valid_task_numbers = self.image_tasks[the_image_id]

        some_coco = self.task_cocos[valid_task_numbers[0]]

        the_image_dict = some_coco.loadImgs(the_image_id)[0]
        I = load_image(get_image_file_name(the_image_dict))

        all_image_anns = some_coco.loadAnns(some_coco.getAnnIds(imgIds=the_image_id))

        if len(all_image_anns) < MAX_GPU_SIZE:
            selected_image_anns = all_image_anns
        else:
            selected_image_anns = np.random.choice(all_image_anns, size=MAX_GPU_SIZE)

        B = len(selected_image_anns)

        x = [
            image_transforms(crop_img_to_bbox(I, target_transforms(a["bbox"], I.size)))
            for a in selected_image_anns
        ]

        bbox = get_bbox_array_from_annotations(selected_image_anns, I.size)

        t = np.zeros((B, len(TASK_NUMBERS)), dtype=np.float32)
        m = np.zeros(len(TASK_NUMBERS), dtype=np.uint8)
        c = get_one_hot(
            [(a["COCO_category_id"] - 1) for a in selected_image_anns], num_classes=90
        )
        d = np.ones((B, 1), dtype=np.float32)

        for task_number in valid_task_numbers:
            task_anns = self.task_cocos[task_number].loadAnns(
                [a["id"] for a in selected_image_anns]
            )

            task_id = task_number_to_task_id(task_number)

            for i, a in enumerate(task_anns):
                if a["category_id"] == 1:
                    t[i, task_id] = 1

            m[task_id] = 1

        return (
            default_collate(x),
            torch.tensor(t).float(),
            torch.tensor(m).byte(),
            torch.tensor(c).float(),
            torch.tensor(d).float(),
            torch.tensor(bbox).float(),
        )

    @staticmethod
    def custom_collate(batch):
        x, t, m, c, d, bbox = batch[0]
        return x, t, m, c, d

    @staticmethod
    def custom_collate_with_bbox(batch):
        x, t, m, c, d, bbox = batch[0]
        return x, t, m, c, d, bbox


class CocoTasksGT(Dataset):
    def __init__(self, task_number: int, set_name: str):
        assert task_number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        assert set_name in ["train"]
        self.len_lambda = 60

        self.task_number = task_number
        self.set_name = set_name

        self.annotation_file = os.path.join(
            COCO_TASKS_ANNOTATIONS_ROOT,
            "task_{}_{}.json".format(self.task_number, self.set_name),
        )

        with redirect_stdout(devnull):
            self.task_coco = COCO(self.annotation_file)

        self.list_of_valid_images = []
        for image_id in self.task_coco.getImgIds():
            if len(self.task_coco.getAnnIds(imgIds=image_id)) > 0:
                self.list_of_valid_images.append(image_id)

    def __len__(self) -> int:
        return len(self.list_of_valid_images)

    def __repr__(self) -> str:
        return "{dbname}({tname}-{sname})".format(
            dbname=self.__class__.__name__, tname=self.task_number, sname=self.set_name
        )

    def __getitem__(
        self, item: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[dict]]:
        """
        Give me an item. An item is basically the annotation of an image.
        :param item: the index of the item to get
        :return: tuple (x, c, d, t, detections),
            x is the transformed image data for different boxes, dim: [B x 3 x 224x 224]
            c is the one hot encoding of the classes, dim: [B x num_classes]
            d is the detection scores for those boxes, dim: [B x 1]
            t is the target which shows whether each of those boxes are first choice or not, dim: [B x 1]
            detections is the json file containing the raw detections, type: List[dict]
        """
        the_image_id = self.list_of_valid_images[item]
        the_image_dict = self.task_coco.loadImgs(the_image_id)[0]

        preferred_anns = self.task_coco.loadAnns(
            self.task_coco.getAnnIds(imgIds=the_image_id, catIds=1)
        )
        non_preferred_anns = self.task_coco.loadAnns(
            self.task_coco.getAnnIds(imgIds=the_image_id, catIds=0)
        )

        I = load_image(get_image_file_name(the_image_dict))

        anns = []
        t = []

        # what should be the size?
        size = int(round(np.random.poisson(self.len_lambda)))

        size += 1  # ensure size is larger than 1
        if size > MAX_GPU_SIZE:  # ensure that size is smaller than MAX_GPU_SIZE
            size = MAX_GPU_SIZE

        # add from preferred_anns
        number_of_preferred = len(preferred_anns)
        if number_of_preferred > 0:
            which_preferred_to_add = np.random.choice(
                preferred_anns, size=number_of_preferred, replace=False
            )

            anns.extend([a for a in which_preferred_to_add])
            t.extend([1] * number_of_preferred)
            size -= number_of_preferred

        if size > 0:
            # add non preferred anns
            number_of_non_preferred = len(non_preferred_anns)
            if number_of_non_preferred > size:
                number_of_non_preferred = size

            # this is the maximum
            number_of_non_preferred = min(
                number_of_non_preferred, len(non_preferred_anns)
            )

            if number_of_non_preferred > 0:
                which_non_preferred_to_add = np.random.choice(
                    non_preferred_anns, size=number_of_non_preferred, replace=False
                )

                anns.extend([a for a in which_non_preferred_to_add])
                t.extend([0] * number_of_non_preferred)
                size -= number_of_non_preferred

        # choose detections for annotation ids
        detections = [
            {"bbox": a["bbox"], "score": 1.0, "category_id": a["COCO_category_id"]}
            for a in anns
        ]

        # fill in cropped images
        x = [
            image_transforms(
                crop_img_to_bbox(I, target_transforms(det["bbox"], I.size))
            )
            for det in detections
        ]

        c = get_one_hot([(a["category_id"] - 1) for a in detections], num_classes=90)

        # fill in the detection scores
        d = [float(det["score"]) for det in detections]

        # unsqueeze to make [B] -> [B x 1]
        return (
            default_collate(x),
            Tensor(c).float(),
            Tensor(d).unsqueeze(1),
            Tensor(t).unsqueeze(1),
            detections,
        )

    @staticmethod
    def custom_collate(batch):
        x, c, d, t, detections = batch[0]
        return x, c, d, t, detections


class CocoTasksTestGT(Dataset):
    def __init__(self, task_number: int):
        assert task_number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        self.task_number = task_number
        self.annotation_file = os.path.join(
            COCO_TASKS_ANNOTATIONS_ROOT, "task_{}_test.json".format(self.task_number)
        )

        with redirect_stdout(devnull):
            self.task_coco = COCO(self.annotation_file)

        self.list_of_valid_images = []
        for image_id in self.task_coco.getImgIds():
            if len(self.task_coco.getAnnIds(imgIds=image_id)) > 0:
                self.list_of_valid_images.append(image_id)

    def __len__(self) -> int:
        return len(self.list_of_valid_images)

    def __repr__(self) -> str:
        return "{dbname}({tname})".format(
            dbname=self.__class__.__name__, tname=self.task_number
        )

    def __getitem__(self, item) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[dict]]:
        """
        :param item:
        :return: tuple (x, c, d, bbox, detections)
        """
        the_image_id = self.list_of_valid_images[item]
        the_image_dict = self.task_coco.loadImgs(the_image_id)[0]

        related_annotations = self.task_coco.loadAnns(
            self.task_coco.getAnnIds(imgIds=the_image_id)
        )

        detections = [
            {
                "bbox": a["bbox"],
                "score": 1.0,
                "category_id": a["COCO_category_id"],
                "image_id": the_image_id,
            }
            for a in related_annotations
        ]

        if len(detections) > MAX_GPU_SIZE * 2:
            print("WARNING, test image num detection larger than MAX_GPU_SIZE")

        # TODO: fix this later
        detections = detections[: MAX_GPU_SIZE * 2]

        I = load_image(get_image_file_name(the_image_dict))

        # fill in cropped images
        x = [
            image_transforms(
                crop_img_to_bbox(I, target_transforms(det["bbox"], I.size))
            )
            for det in detections
        ]

        bbox = get_bbox_array_from_annotations(detections, I.size)

        c = get_one_hot([(a["category_id"] - 1) for a in detections], num_classes=90)

        # fill in the detection scores
        d = [float(det["score"]) for det in detections]

        return (
            default_collate(x),
            torch.tensor(c).float(),
            Tensor(d).unsqueeze(1),
            torch.tensor(bbox).float(),
            detections,
        )

    @staticmethod
    def custom_collate(batch):
        x, c, d, bbox, detections = batch[0]
        return x, c, d, detections

    @staticmethod
    def custom_collate_with_bbox(batch):
        x, c, d, bbox, detections = batch[0]
        return x, c, d, bbox, detections


class CocoTasksTest(Dataset):
    THRESH = 0.02

    def __init__(self, task_number: int):
        assert task_number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        self.task_number = task_number
        self.annotation_file = os.path.join(
            COCO_TASKS_ANNOTATIONS_ROOT, "task_{}_test.json".format(self.task_number)
        )

        with redirect_stdout(devnull):
            self.task_coco = COCO(self.annotation_file)

        detections_file = COCO_TASKS_TEST_DETECTIONS
        with open(detections_file) as f:
            self.detections = json.load(f)

        related_detections = []
        for d in self.detections:
            if d["score"] > self.THRESH:
                related_detections.append(d)

        self.detections = related_detections

        self.per_image_detections = defaultdict(list)
        for d in self.detections:
            self.per_image_detections[d["image_id"]].append(d)

        self.list_of_valid_images = []
        for image_id in self.task_coco.getImgIds():
            if len(self.per_image_detections[image_id]) > 0:
                self.list_of_valid_images.append(image_id)

    def __len__(self) -> int:
        return len(self.list_of_valid_images)

    def __repr__(self) -> str:
        return "{dbname}({tname})".format(
            dbname=self.__class__.__name__, tname=self.task_number
        )

    def __getitem__(self, item) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[dict]]:
        """
        :param item:
        :return: tuple (x, c, d, bbox, detections)
        """
        the_image_id = self.list_of_valid_images[item]
        the_image_dict = self.task_coco.loadImgs(the_image_id)[0]

        detections = self.per_image_detections[the_image_id]

        detections = sorted(detections, key=lambda i: i["score"], reverse=True)

        if len(detections) > MAX_GPU_SIZE * 2:
            print("WARNING, test image num detection larger than MAX_GPU_SIZE")

        # TODO: fix this later
        detections = detections[: MAX_GPU_SIZE * 2]

        I = load_image(get_image_file_name(the_image_dict))

        # fill in cropped images
        x = [
            image_transforms(
                crop_img_to_bbox(I, target_transforms(det["bbox"], I.size))
            )
            for det in detections
        ]

        bbox = get_bbox_array_from_annotations(detections, I.size)

        c = get_one_hot([(a["category_id"] - 1) for a in detections], num_classes=90)

        # fill in the detection scores
        d = [float(det["score"]) for det in detections]

        return (
            default_collate(x),
            torch.tensor(c).float(),
            Tensor(d).unsqueeze(1),
            torch.tensor(bbox).float(),
            detections,
        )

    @staticmethod
    def custom_collate(batch):
        x, c, d, bbox, detections = batch[0]
        return x, c, d, detections

    @staticmethod
    def custom_collate_with_bbox(batch):
        x, c, d, bbox, detections = batch[0]
        return x, c, d, bbox, detections
