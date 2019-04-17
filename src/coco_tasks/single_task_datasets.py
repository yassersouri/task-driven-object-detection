import json
import os
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Tuple, List

import numpy as np
from PIL import Image
from boltons.cacheutils import LRU
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# noinspection PyUnresolvedReferences
from coco_tasks import _init_path
from coco_tasks import transforms
from coco_tasks.settings import (
    TASK_NUMBERS,
    COCO_TRAIN_IMAGES,
    COCO_VAL_IMAGES,
    COCO_TASKS_ANNOTATIONS_ROOT,
    COCO_TASKS_TEST_DETECTIONS,
)
from pycocotools.coco import COCO

MAX_GPU_SIZE = 64
IMAGE_CACHE = LRU(max_size=100)
devnull = open(os.devnull, "w")


def load_image(file_name: str) -> Image.Image:
    if file_name in IMAGE_CACHE:
        image = IMAGE_CACHE.get(file_name)
    else:
        image = Image.open(file_name)
        image = image.convert("RGB")
        IMAGE_CACHE[file_name] = image
    return image


def crop_img_to_bbox(img: Image.Image, bbox: List[float]) -> Image.Image:
    """
    This function does zero padding if the bbox goes over the edges of the image.
    """
    cropped_image = img.crop((bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]))
    return cropped_image


def _image_name_to_image_root(image_name: str) -> str:
    if image_name.startswith("COCO_train2014"):
        return COCO_TRAIN_IMAGES
    elif image_name.startswith("COCO_val2014"):
        return COCO_VAL_IMAGES
    else:
        raise Exception("Not recognizable set from image_name", image_name)


def get_image_file_name(img: dict) -> str:
    image_root = _image_name_to_image_root(img["file_name"])
    return os.path.join(image_root, img["file_name"])


image_transforms = Compose(
    [
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

target_transforms = transforms.BBoxCompose(
    [transforms.MakeBBoxSquare(), transforms.ScaleAwarePadding(padding=0.1)]
)


class CocoTasksGT(Dataset):
    def __init__(self, task_number: int, set_name: str):
        assert task_number in TASK_NUMBERS
        assert set_name in ["train"]
        self.len_lambda = 60

        self.task_number = task_number
        self.set_name = set_name
        self.only_relevant = False

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
        return "{dbname}({tname}-{sname}-or:{r})".format(
            dbname=self.__class__.__name__,
            tname=self.task_number,
            sname=self.set_name,
            r=self.only_relevant,
        )

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor, List[dict]]:
        """
        Give me an item. An item is basically the annotation of an image.
        :param item: the index of the item to get
        :return: tuple (x, d, t, detections),
            x is the transformed image data for different boxes, dim: [B x 3 x 224x 224]
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

        # fill in the detection scores
        d = [float(det["score"]) for det in detections]

        # unsqueeze to make [B] -> [B x 1]
        return (
            default_collate(x),
            Tensor(d).unsqueeze(1),
            Tensor(t).unsqueeze(1),
            detections,
        )

    @staticmethod
    def custom_collate(batch):
        x, t, d, detections = batch[0]
        return x, t, d, detections


class CocoTasksTest(Dataset):
    THRESH = 0.02

    def __init__(self, task_number: int):
        assert task_number in TASK_NUMBERS

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

    def __getitem__(self, item) -> Tuple[Tensor, Tensor, List[dict]]:
        """
        :param item:
        :return: tuple (x, d, detections)
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

        # fill in the detection scores
        d = [float(det["score"]) for det in detections]

        return default_collate(x), Tensor(d).unsqueeze(1), detections

    @staticmethod
    def custom_collate(batch):
        x, d, detections = batch[0]
        return x, d, detections


class CocoTasksTestGT(Dataset):
    def __init__(self, task_number: int):
        assert task_number in TASK_NUMBERS

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

    def __getitem__(self, item) -> Tuple[Tensor, Tensor, List[dict]]:
        """
        :param item:
        :return: tuple (x, d, detections)
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

        # fill in the detection scores
        d = [float(det["score"]) for det in detections]

        return default_collate(x), Tensor(d).unsqueeze(1), detections

    @staticmethod
    def custom_collate(batch):
        x, d, detections = batch[0]
        return x, d, detections
