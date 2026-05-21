# hawk-eye
> Code to find targets in aerial imagery.

![Python3 CI](https://github.com/uavaustin/hawk-eye/workflows/Python3%20CI/badge.svg)

----

`hawk-eye` is an end-to-end pipeline for training models to detect aerial imagery.
Specifically, the goal is to produce models that perform well in the [AUVSI SUAS](https://static1.squarespace.com/static/5d554e14aaa5e300011a4844/t/5fd3780f2662933f59dbedd5/1607694352554/auvsi_suas-2021-rules.pdf)
competition.

We train our models with a fusion of real and synthetic data and package our inferencing
pipeline as a python pip package. The output of this project is utilized by UAV Austin's
[Orchestra](https://github.com/uavaustin/orchestra) team.


## Overview

![overview](https://user-images.githubusercontent.com/31543169/111933494-98973400-8a8d-11eb-930f-38fe56573c82.gif)


## Contents

* [`Setup`](#setup)
* [`Data Generation`](#data-generation)
* [`Model Training`](#model-training)
* [`Inference`](#inference)
* [`Testing`](#testing)
* [`Bazel`](#bazel)
* [`Style`](#style)


## Setup
This project supports Linux, WSL, and Mac systems. To setup one of these environments, run:

`hawk_eye/setup/setup_env.sh`

This file also takes an optional argument to a python virtual environment:

`hawk_eye/setup/setup_env.sh ~/path_to_venv`

Upon sucessful termination of this script, you need to then recieve access to Google
Cloud Storage. See your lead about gaining permissions, then run

`hawk_eye/setup/install_google_cloud.sh`

To ensure you are all setup and ready to code, you can test code by running:

`bazel test //...`


## Data Generation

Before we can ever train a model we need data. In our project, we actually create
synthetic data using various python libraries. Inside of `hawk_eye/data_generation`
you'll find the scripts related to data processing and more information.


## Model Training

Currently we train both classifiers and detection models. This is in flux as the pace
of discovery in machine learning is rapid.

See `hawk_eye/train/README.md` for more information on how to running training jobs.


## Inference

Inference on an image will utilize both the classifier and object detector models.
The inference script can be run as follows:

```
PYTHONPATH=$(pwd) inference/find_targets.py \
    --image_path /path/to/image \
    --visualization_dir /path/to/save
```
The command above will visualize and save the models' predictions. See
`inference/find_targets.py` for the full list of available arguments.

One can also specify the model timestamps if you have a certain model to test.


## Testing

Testing is done with `bazel`. Please see [the docs](https://uavaustin.github.io/hawk-eye/tests.html)
for more information on writing and running tests.


## Bazel

`Bazel` is an open-sourced Google project used for a variety of build environments. Right
now, we mainly use it for python testing, but if PyTorch eventually supports building
itself as an external third party project, we might start using the C++ PyTorch API.

For now, if you're interested,
[here](https://docs.bazel.build/versions/master/user-manual.html)
is some documentation.

`hawk_eye` closely models the repository structure of Google's [`MediaPipe`](https://github.com/google/mediapipe).
Please see this project when you have questions about how to structure certain Bazel
files.

## Style

Python code is formatted with `Black` and import-sorted with `isort` through
`pre-commit`. CI also runs `flake8`, the repository strict lint gate, and the
quality gates in `tools/quality_gates.py`. Please familiarize yourself with the
[`Google python style guide`](https://google.github.io/styleguide/pyguide.html).
