# hawk_eye.core.detector

## Overview
UAV uses object detectors to predict the shape type of the AUVSI SUAS targets.
Currently, the detectors are defined by the RetinaNet architecrure with a
backbone, FPN, RetinaNet head, and anchor box regression. This architecture
works accross a variety of tasks, including ours, but the architecture can be
daunting to beginners.

Newer models like FCOS, CenterNet, and Foveabox, are a class of anchor-less
detectors which are simpler architectures. These models would likely work well
on our project as well and are worth considering as alternatives.

For UAV, the targets are composed of a shape and alphanumeric (A-Z or 0-9).
Due to the limitations of our camera, we somtimes lack the resolution for a
model to determine the alphanumeric (even a human sometimes can't). This is
the motivating factor for why the object detectors we run only predict the
shape class of the target.

## Module
```{todo}
Write function docs.
```
```{automodule} hawk_eye.core.detector
:members:
```
