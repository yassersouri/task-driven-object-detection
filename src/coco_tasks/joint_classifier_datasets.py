import os
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from coco_tasks.settings import TASK_NUMBERS, COCO_TASKS_ANNOTATIONS_ROOT
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


def task_number_to_task_id(task_number: int) -> int:
    return TASK_NUMBERS.index(task_number)


def task_id_to_task_number(task_id: int) -> int:
    return TASK_NUMBERS[task_id]


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

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Gives the annotations of an image for all tasks. Also generates appropriate masks if the image
        doesn't have annotations for a particular image.
        :param item: the index of the image.
        :return: tuple(x, t, m),
            x is the transformed image data for different boxes, bim: [B x 3 x 224 x 224]
            t is the target. t[i, tn] is 1 if the image i i first choice for task tn. Otherwise it is 0. [B x task_numbers]
            m is the mask. m[tn] is 1 if this image is annotated for task tn. Otherwise it is 0. [task_numbers]
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

        t = np.zeros((B, len(TASK_NUMBERS)), dtype=np.float32)
        m = np.zeros(len(TASK_NUMBERS), dtype=np.uint8)

        for task_number in valid_task_numbers:
            task_anns = self.task_cocos[task_number].loadAnns(
                [a["id"] for a in selected_image_anns]
            )

            task_id = task_number_to_task_id(task_number)

            for i, a in enumerate(task_anns):
                if a["category_id"] == 1:
                    t[i, task_id] = 1

            m[task_id] = 1

        return default_collate(x), torch.tensor(t).float(), torch.tensor(m).byte()

    @staticmethod
    def custom_collate(batch):
        x, t, m = batch[0]
        return x, t, m
