# Overview

The `hawk_eye` inference application uses a combination of binary classification
and object detectors to find our targets.

Here is a gist of the pipeline:

    1. Recieve an image from the plane's camera. Typically these images have too many
       pixels for a model to efficiently look at in one go. So, the image is slices into
       smaller squares with a known size (e.g. 512x512 pixels).

    2. Send the smaller images through the classification network and get back whether
       or not each tile contains a target or is a background.

    3. For images with targets in them, then these through the object detector to find
       out which target shape is present and where it is in the image.

    4. With a known location of the target in the smaller image, we transform the
       target's location into the coordinate space of the original image.

## Future Work

We need to also determine the alphanumeric present on the target and the colors of the
alphanumeric and shape. This can be challenging when the alphanumeric is barely visible.
