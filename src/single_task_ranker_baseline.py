import json
import os
import random
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Dict, List

import click
import numpy as np
import torch
from boltons.fileutils import mkdir_p
from torch.optim import Adam

from coco_tasks.ranker_datasets import (
    CocoTasksRanker,
    CocoTasksRankerTestGT,
    CocoTasksRankerTest,
)
from coco_tasks.ranker_experiments import Experiment
from coco_tasks.ranker_networks import Ranker
from coco_tasks.settings import SAVING_DIRECTORY
from pycocotools.cocoeval import COCOeval

THRESH = 0.1


def top_k_fuse(
    image_detections: List[dict],
    max_detections_num: int,
    detection_score_threshold: float,
) -> List[dict]:
    image_detections = list(
        filter(lambda d: d["score"] >= detection_score_threshold, image_detections)
    )

    sorted_detections = list(
        sorted(image_detections, key=lambda x: x["rank_value"], reverse=True)
    )

    fused_detections = []
    for i, d in enumerate(sorted_detections):
        the_r = i + 1
        the_detection = d
        the_detection["p"] = 1.0 - float((the_r - 1) / max_detections_num)
        fused_detections.append(the_detection)

    return fused_detections


def fuse(detections_per_image: Dict[int, List[dict]]) -> List[dict]:
    results_detections = []

    max_detections_num = max(
        [len(detections_per_image[iid]) for iid in detections_per_image.keys()]
    )

    for image_id in detections_per_image.keys():
        fused_detections = top_k_fuse(
            detections_per_image[image_id], max_detections_num, THRESH
        )
        # noinspection PyUnboundLocalVariable
        for f in fused_detections:
            to_append = {
                "image_id": f["image_id"],
                "category_id": 1,
                "score": f["p"],
                "bbox": f["bbox"],
                "original_category": f["category_id"],
            }

            results_detections.append(to_append)

    return results_detections


@click.command()
@click.option("--random-seed", envvar="SEED", default=0)
@click.option("--task-number", type=int)
@click.option("--test-on-gt", type=bool, default=False)
@click.option("--only-test", type=bool, default=False)
@click.option("--overfit", type=bool, default=False)
def main(random_seed, task_number, test_on_gt, only_test, overfit):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    n_epochs = 3
    lr = 1e-4
    wd = 0

    train_db = CocoTasksRanker(task_number)

    if test_on_gt:
        test_db = CocoTasksRankerTestGT(task_number)
    else:
        test_db = CocoTasksRankerTest(task_number)

    network = Ranker()
    optimizer = Adam(network.parameters(), lr=lr, weight_decay=wd)
    experiment = Experiment(network, train_db, optimizer=optimizer, tensorboard=True)

    folder = "single-task-ranker-baseline-tn:{tn}-seed:{s}".format(
        tn=task_number, s=random_seed
    )

    folder = os.path.join(SAVING_DIRECTORY, folder)
    mkdir_p(folder)

    if not only_test:
        # train
        experiment.train_n_epochs(n_epochs, lr_scheduler=True, overfit=overfit)

        # save model
        torch.save(network.state_dict(), os.path.join(folder, "model.mdl"))
    else:
        # load model
        network.load_state_dict(torch.load(os.path.join(folder, "model.mdl")))

    # test model
    detections = experiment.do_test(test_db)

    # save detections
    with open(
        os.path.join(folder, "detections_teg:{teg}.json".format(teg=test_on_gt)), "w"
    ) as f:
        json.dump(detections, f)

    detections_per_image = defaultdict(list)
    for d in detections:
        detections_per_image[d["image_id"]].append(d)

    fusion = "top_k"

    fused_detections = fuse(detections_per_image=detections_per_image)

    with open(
        os.path.join(
            folder, "detections_teg:{teg}_f:{f}.json".format(teg=test_on_gt, f=fusion)
        ),
        "w",
    ) as f:
        json.dump(fused_detections, f)

    # perform evaluation
    with redirect_stdout(open(os.devnull, "w")):
        gtCOCO = test_db.task_coco
        dtCOCO = gtCOCO.loadRes(
            os.path.join(
                folder,
                "detections_teg:{teg}_f:{f}.json".format(teg=test_on_gt, f=fusion),
            )
        )
        cocoEval = COCOeval(gtCOCO, dtCOCO, "bbox")
        cocoEval.params.catIds = 1
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    print("fusion: %s - mAP:\t\t %1.6f" % (fusion, cocoEval.stats[0]))
    print("fusion: %s - ap@.5:\t\t %1.6f" % (fusion, cocoEval.stats[1]))

    # save evaluation performance
    with open(
        os.path.join(
            folder, "result_teg:{teg}_f:{f}.json".format(teg=test_on_gt, f=fusion)
        ),
        "w",
    ) as f:
        f.write("%1.6f, %1.6f" % (cocoEval.stats[0], cocoEval.stats[1]))


if __name__ == "__main__":
    main()
