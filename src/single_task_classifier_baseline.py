import json
import os
import random
from contextlib import redirect_stdout

import click
import numpy as np
import torch
from boltons.fileutils import mkdir_p
from torch.optim import SGD

from coco_tasks.settings import SAVING_DIRECTORY
from coco_tasks.single_task_datasets import CocoTasksGT, CocoTasksTest, CocoTasksTestGT
from coco_tasks.single_task_experiments import ClassifierExperiment
from coco_tasks.single_task_networks import ClassifierBaselineNetwork
from pycocotools.cocoeval import COCOeval


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
    lr = 1e-2
    wd = 0

    train_db = CocoTasksGT(task_number, "train")

    if test_on_gt:
        test_db = CocoTasksTestGT(task_number)
    else:
        test_db = CocoTasksTest(task_number)

    network = ClassifierBaselineNetwork()
    optimizer = SGD(network.parameters(), lr=lr, weight_decay=wd)
    experiment = ClassifierExperiment(network, train_db, optimizer=optimizer)

    folder = "single-task-classifier-baseline-tn:{tn}-seed:{s}".format(
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

    # test_model
    detections = experiment.do_test(test_db)

    # save detections
    with open(
        os.path.join(folder, "detections-teg:{teg}.json".format(teg=test_on_gt)), "w"
    ) as f:
        json.dump(detections, f)

    # perform evaluation
    with redirect_stdout(open(os.devnull, "w")):
        gtCOCO = test_db.task_coco
        dtCOCO = gtCOCO.loadRes(
            os.path.join(folder, "detections-teg:{teg}.json".format(teg=test_on_gt))
        )
        cocoEval = COCOeval(gtCOCO, dtCOCO, "bbox")
        cocoEval.params.catIds = 1
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    print("mAP:\t\t %1.6f" % cocoEval.stats[0])
    print("ap@.5:\t\t %1.6f" % cocoEval.stats[1])

    # save evaluation performance
    with open(
        os.path.join(folder, "result-teg:{teg}.txt".format(teg=test_on_gt)), "w"
    ) as f:
        f.write("%1.6f, %1.6f" % (cocoEval.stats[0], cocoEval.stats[1]))


if __name__ == "__main__":
    main()
