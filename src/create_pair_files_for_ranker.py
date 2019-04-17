import os
from contextlib import redirect_stdout
from itertools import combinations, product
from pickle import dump

# noinspection PyUnresolvedReferences
from coco_tasks import _init_path
from coco_tasks.settings import (
    COCO_TASKS_ANNOTATIONS_ROOT,
    COCO_TASKS_PAIR_ROOT,
    TASK_NUMBERS,
)
from pycocotools.coco import COCO

devnull = open(os.devnull, "w")


def do_stuff(task_number):
    pairs_file_path = os.path.join(
        COCO_TASKS_PAIR_ROOT, "pairs_train_{}.pkl".format(task_number)
    )
    with redirect_stdout(devnull):
        task_coco = COCO(
            os.path.join(
                COCO_TASKS_ANNOTATIONS_ROOT,
                "task_{}_{}.json".format(task_number, "train"),
            )
        )
    pairs = []
    for i, img_id in enumerate(task_coco.getImgIds()):
        all_anns = task_coco.loadAnns(task_coco.getAnnIds(imgIds=img_id))

        first_choice_anns = list(filter(lambda x: x["category_id"] == 1, all_anns))
        non_first_choice_anns = list(filter(lambda x: x["category_id"] == 0, all_anns))
        has_first_choice = len(first_choice_anns) > 0

        if has_first_choice:
            if len(first_choice_anns) > 1:
                for fc1, fc2 in combinations(first_choice_anns, 2):
                    pairs.append((img_id, fc1["id"], fc2["id"], 0.5))
            if len(non_first_choice_anns) > 0:
                for fc, nfc in product(first_choice_anns, non_first_choice_anns):
                    pairs.append((img_id, fc["id"], nfc["id"], 1.0))

    with open(pairs_file_path, "wb") as f:
        dump(pairs, f)


def main():
    for task_number in TASK_NUMBERS:
        do_stuff(task_number)


if __name__ == "__main__":
    main()
