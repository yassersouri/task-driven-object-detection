import json
import os
import random
from contextlib import redirect_stdout

import click
import numpy as np
import torch
from boltons.fileutils import mkdir_p
from torch.optim import SGD

from coco_tasks.graph_datasets import JointCocoTasks, CocoTasksTest, CocoTasksTestGT
from coco_tasks.graph_experiments import JointGraphExperiment
from coco_tasks.graph_networks import AllLinearAggregatorWeightedWithDetScore
from coco_tasks.graph_networks import (
    GGNN,
    InitializerMul,
    AllLinearAggregator,
    OutputModelFirstLast,
)
from coco_tasks.settings import SAVING_DIRECTORY, TASK_NUMBERS
from pycocotools.cocoeval import COCOeval


@click.command()
@click.option("--random-seed", envvar="SEED", default=0)
@click.option("--test-on-gt", type=bool, default=True)
@click.option("--only-test", type=bool, default=False)
@click.option("--overfit", type=bool, default=False)
@click.option("--weighted-aggregation", type=bool, default=False)
def main(random_seed, test_on_gt, only_test, overfit, weighted_aggregation):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    n_epochs = 3
    lr = 1e-2
    wd = 0
    lr_scheduler = True

    # graph settings
    h_dim = 128
    x_dim = 128
    c_dim = 90
    phi_dim = 2048

    if only_test:
        train_db = None
    else:
        train_db = JointCocoTasks()

    initializer = InitializerMul(h_dim=h_dim, phi_dim=phi_dim, c_dim=c_dim)
    if weighted_aggregation:
        aggregator = AllLinearAggregatorWeightedWithDetScore(
            in_features=h_dim, out_features=x_dim
        )
    else:
        aggregator = AllLinearAggregator(in_features=h_dim, out_features=x_dim)
    output_model = OutputModelFirstLast(h_dim=h_dim, num_tasks=len(TASK_NUMBERS))

    network = GGNN(
        initializer=initializer,
        aggregator=aggregator,
        output_model=output_model,
        h_dim=h_dim,
        x_dim=x_dim,
        class_dim=c_dim,
    )
    optimizer = SGD(network.parameters(), lr=lr, weight_decay=wd)
    experiment = JointGraphExperiment(
        network=network,
        optimizer=optimizer,
        dataset=train_db,
        tensorboard=True,
        seed=random_seed,
    )

    train_folder = "ablation-ggnn-seed:{s}".format(s=random_seed)

    folder = os.path.join(SAVING_DIRECTORY, train_folder)
    mkdir_p(folder)

    if not only_test:
        experiment.train_n_epochs(n_epochs, overfit=overfit, lr_scheduler=lr_scheduler)

        torch.save(network.state_dict(), os.path.join(folder, "model.mdl"))
    else:
        network.load_state_dict(torch.load(os.path.join(folder, "model.mdl")))

    for task_number in TASK_NUMBERS:
        if test_on_gt:
            test_db = CocoTasksTestGT(task_number)
        else:
            test_db = CocoTasksTest(task_number)

        print("testing task {}".format(task_number), "---------------------")

        # test_model
        detections = experiment.do_test(test_db, task_number=task_number)

        detections_file_name = "detections_wa:{}_tn:{}_tgt:{}.json".format(
            weighted_aggregation, task_number, test_on_gt
        )

        # save detections
        with open(os.path.join(folder, detections_file_name), "w") as f:
            json.dump(detections, f)

        # perform evaluation
        with redirect_stdout(open(os.devnull, "w")):
            gtCOCO = test_db.task_coco
            dtCOCO = gtCOCO.loadRes(os.path.join(folder, detections_file_name))
            cocoEval = COCOeval(gtCOCO, dtCOCO, "bbox")
            cocoEval.params.catIds = 1
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

        print("mAP:\t\t %1.6f" % cocoEval.stats[0])
        print("ap@.5:\t\t %1.6f" % cocoEval.stats[1])

        # save evaluation performance
        result_file_name = "result_wa:{}_tn:{}_tgt:{}.txt".format(
            weighted_aggregation, task_number, test_on_gt
        )

        with open(os.path.join(folder, result_file_name), "w") as f:
            f.write("%1.6f, %1.6f" % (cocoEval.stats[0], cocoEval.stats[1]))


if __name__ == "__main__":
    main()
