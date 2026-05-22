# hawk_eye.core.classifier

## Overview
Image classification aims to assign a class to an image. For UAV, we are focued
on determining if an image is either "background" or "target." A background image
does not contain one of the AUVSI SUAS targets that we are interested. A image
classified as a target should have one of these targets. Note, in this model, we
don't actually determine the specific target, that is left to a detector. In this
way, the classifier works as a filter to feed only taget-containing images into
the object detector.

UAV Austin uses a fast classifier to speed up processing time, saving more room
for the slower object detector. As technology improves, it might be possible to
remove the classifier from the step completely and solely rely on the detector to
process each image.

## Module
```{automodule} hawk_eye.core.classifier
:members:
```
