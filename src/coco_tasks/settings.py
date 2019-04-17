import os

COCO_ROOT = os.path.join(os.path.expanduser("~"), "mscoco")
COCO_TRAIN_IMAGES = os.path.join(COCO_ROOT, "train2014")
COCO_VAL_IMAGES = os.path.join(COCO_ROOT, "val2014")

COCO_TASKS_ROOT = os.path.join(COCO_ROOT, "coco-tasks")
COCO_TASKS_ANNOTATIONS_ROOT = os.path.join(COCO_TASKS_ROOT, "annotations")
COCO_TASKS_PAIR_ROOT = os.path.join(COCO_TASKS_ROOT, "pairs")
COCO_TASKS_TEST_DETECTIONS = os.path.join(COCO_TASKS_ROOT, "detections_faster.json")

TB_ROOT = os.path.join(COCO_TASKS_ROOT, "TB")
SAVING_DIRECTORY = os.path.join(COCO_TASKS_ROOT, "saving_directory")

TASK_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

for d in [SAVING_DIRECTORY, COCO_TASKS_PAIR_ROOT]:
    os.makedirs(d, exist_ok=True)
