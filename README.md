# Task Driven Object Detection

Author's implementation of the paper ["What Object Should I Use? - Task Driven Object Detection"](https://arxiv.org/abs/1904.03000).

If you use our code, please cite our paper.

```latex
@inproceedings{cocotasks2019,
    Author    = {Sawatzky, Johann and Souri, Yaser and Grund, Christian and Gall, Juergen},
    Title     = {{What Object Should I Use? - Task Driven Object Detection}},
    Booktitle = {{CVPR}},
    Year      = {2019}
}
```

Table of Contents
=================
  * [Requirements](#requirements)
     * [Installing COCO API](#installing-coco-api)
  * [Getting The Data](#getting-the-data)
     * [COCO Dataset](#coco-dataset)
     * [COCO-Tasks Dataset](#coco-tasks-dataset)
     * [Final Directory Structure](#final-directory-structure)
  * [Reproducing Results](#reproducing-results)
     * [General Information](#general-information)
        * [Settings](#settings)
        * [Seeds](#seeds)
        * [Running on Detections vs. on Ground Truth Bounding Boxes](#running-on-detections-vs-on-ground-truth-bounding-boxes)
        * [Changing the Detector](#changing-the-detector)
     * [Baselines](#baselines)
        * [1. Classifier Baseline](#1-classifier-baseline)
        * [2. Ranker Baseline](#2-ranker-baseline)
     * [Ablation Experiments and Proposed Method](#ablation-experiments-and-proposed-method)
        * [1. Ablation: Joint Classifier](#1-ablation-joint-classifier)
        * [2. Ablation: Joint Classifier   Class](#2-ablation-joint-classifier--class)
        * [3. Ablation: Joint GGNN   Class (  Weighted Aggregation)](#3-ablation-joint-ggnn--class--weighted-aggregation)
        * [4. Proposed Method (  Fusion)](#4-proposed-method--fusion)
  * [Bugs, Feedback and Questions](#bugs-feedback-and-questions)

## Requirements

The two main dependencies are:

* Python 3.6
* PyTorch 1.0.x

Other dependencies are listed in the `requirements.txt` file.

### Installing COCO API

There is only 1 special dependency and that is COCO's Python API. The code for the API is already included in the repository. You just need to do the following to compile it.

```bash
cd src/external/coco/PythonAPI
make
```

## Getting The Data

### COCO Dataset

We need the train and validation COCO 2014 dataset. These can be downloaded from [here](http://cocodataset.org/#download). Or you can run the following commands to download them.

```bash
mkdir ~/mscoco  # I assume your main directory will be at your home directory.
# what I usually do is, I symlink ~/mscoco to a place that has enough space.
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip -d ~/mscoco/
unzip val2014.zip -d ~/mscoco/
```

### COCO-Tasks Dataset

Other than the COCO dataset images, you need the COCO-Tasks dataset annotation. These can be downloaded from [here](https://coco-tasks.github.io/). Or you can run the following commands to download them. For the following command to run, you need [Git-LFS](https://git-lfs.github.com/) installed.

```bash
cd ~/mscoco  # The same root directory as above.
git clone -b cvpr2019 --depth 1 git@github.com:coco-tasks/dataset.git coco-tasks
```

### Final Directory Structure

At the end inside `~/mscoco` you should have a directory structure like this:

```
~/mscoco/
├── train2014/
│   ├── COCO_train2014_000000000009.jpg
│   ├── COCO_train2014_000000000025.jpg
│   └── ...
├── val2014/
│   ├── COCO_val2014_000000000042.jpg
│   ├── COCO_val2014_000000000073.jpg
│   └── ...
└── coco-tasks/
    ├── annotations/
	│   ├── task_1_test.json
	│   ├── task_1_train.json
	│   └── ...
	├── image_lists/
    │   ├── imgIds_task_1_test.txt
    │   ├── imgIds_task_1_train.txt
    │   └── ...
    ├── detection_faster.json
    └── ...
```

## Reproducing Results

### General Information

#### Settings

We have a special settings file: `src/coco_tasks/settings.py`.

The main thing that is specified in the settings file is the location of stuff.

Here is a description of the things that you might want to change:

```python
COCO_ROOT  # location of the root folder for data
TB_ROOT  # location that Tensorboard will write its data
SAVING_DIRECTORY  # location where we will save trained models and results
```

#### Seeds

We tried for our results to be reproducible. We set the random seeds.

```python
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
```

But this is not enough. There is still sources (e.g. cuDNN) that introduce stochasticity. To combat this, we run our training and testing 3 times with 3 different random seeds: 0, 1 and 2.

On all of the following training and testing scripts, one can easily set the seed using the `--random-seed 0` command line option. 0 is the default seed.

#### Running on Detections vs. on Ground Truth Bounding Boxes

As mentioned in our paper, we train on ground truth bounding boxes and test in two settings: on detected bounding boxes (by default) and on ground truth bounding boxes (perfect detector setting).

By specifying `--test-on-gt True` you can perform the testing on the ground truth bounding box setting. This command line option works on all of the following python scripts. If you have already once run the code without this option, it means that you have already trained your model and tested it on detected bounding boxes. You can now only perform testing on ground truth bounding boxes by `--test-on-gt True --only-test True`. This will use the already trained models and only performs testing.

#### Changing the Detector

We have trained Faster-RCNN and YOLOv2 of a subset of the COCO images that do not intersect with our test set. We them performed object detection on all images in our test set. The result is provided as part of our dataset release. If you prepared your data as I described above, they should be located at `~/mscoco/coco-tasks/detections_faster.json` and `~/mscoco/coco-tasks/detections_yolo.json`.

By default our code is set to use the Faster-RCNN detector, as specified in the `settings.py` file:

```python
COCO_TASKS_TEST_DETECTIONS = os.path.join(COCO_TASKS_ROOT, "detections_faster.json")
```

You can change that line in the `settings.py` file to use YOLO:

```python
COCO_TASKS_TEST_DETECTIONS = os.path.join(COCO_TASKS_ROOT, "detections_yolo.json")
```

### Baselines

#### 1. Classifier Baseline

This baseline corresponds to 4th row of Table 2 in our paper. Run the following

```bash
python src/single_task_classifier_baseline.py --task-number 1 --random-seed 0
```

This is a single task baseline, which means the model is trained for each task separately.

A good bash script to do all the training and testing would be:

```bash
#!/usr/bin/env bash
task_numbers=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
seeds=(0 1 2)
for i in "${task_numbers[@]}";
do
    for s in "${seeds[@]}";
    do
        python src/single_task_classifier_baseline.py --task-number ${i} --random-seed ${s}
        python src/single_task_classifier_baseline.py --task-number ${i} --random-seed ${s} --test-on-gt True --only-test True
    done
done
```

#### 2. Ranker Baseline

This baseline corresponds to 3rd row of Table 2 in our paper.

This baseline is different from all other baseline and methods in one major way: for training the ranker we have to use pairwise comparisons.

To generate these training pairs you need to first run the following script:

```bash
python src/create_pair_files_for_ranker.py
```

Then you can train the actual baseline:

```bash
python src/single_task_ranker_baseline.py --task-number 1 --seed 0
```

### Ablation Experiments and Proposed Method

#### 1. Ablation: Joint Classifier

This ablation experiment corresponds to 2nd row of Table 3 in our paper: *(a) joint classifier*.

```bash
python src/ablation_joint_classifier.py --random-seed 0
```

There is no need to run this script for tasks one by one, it will train for all tasks at the same time and will test for each task one by one.

#### 2. Ablation: Joint Classifier + Class

This ablation experiment corresponds to 3rd row of Table 3 in our paper: *(b) joint classifier + class*.

```bash
python src/ablation_joint_classifier_withclass.py --random-seed 0
```

#### 3. Ablation: Joint GGNN + Class (+ Weighted Aggregation)

This ablation experiment corresponds to 4th and 5th rows of Table 3 in our paper: *(c) joint GGNN + class* and *(d) joint GGNN + class + w. aggreg.*

Notice that the Weighted Aggregation only has an effect at test time. So if you train a model, you can test it with and without weighted aggregation.

Run the code below to train a model for this ablation experiment and test it without weighted aggregation.

```bash
python src/ablation_ggnn.py --random-seed 0 --weighted-aggregation False
```

Now you can test it with weighted aggregation like this:

```bash
python src/ablation_ggnn.py --random-seed 0 --weighted-aggregation True --only-test True
```

#### 4. Proposed Method (+ Fusion)

This corresponds to our proposed method which are reported in 6th and 7th rows of Table 3 in our paper: *(e) proposed* and *(f) proposed + fusion*.

Notice that fusion only has an effect at test time.

Training a model and testing it without fusion:

```bash
python src/ggnn_full.py --random-seed 0 --fusion none
```

Testing the trained model with fusion:

```bash
python src/ggnn_full.py --random-seed 0 --fusion avg
```



## Bugs, Feedback and Questions

Feel free and open issues on this repository or contact me directly: souri@iai.uni-bonn.de
