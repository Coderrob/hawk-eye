# Overview

To train deep neural nets, we need a lot of data; typically, the more the better.
It's also best practice to have the training data's distribution match the
distribution of the data you want the model to perform well on. In UAV terms, we
want data that accurately depicts the AUVSI SUAS competition targets. Unfortunately,
this data can be hard to come by since there is only one competition per year and
a limited number of flight tests.

To work around this limitation, we have devised a way to create an unbounded number
of artifical training images. The data generation pipeline takes a base shape, perhaps
a circle or square, alters the color, pastes on a letter or number, then pastes the
shape into an aerial image.

A downside to artifical data is it's not a perfect representation of the real targets.
There are some artifacts in the shape edges, predictable color characteristics, among
other things that allow our models to learn quite quickly what is a fake target.

## Future Work

In no particular order of importance, here is a list of future avenues to pursue in
regards to data generation and collection.

- Streamline an approach to take the real targets UAV has to take pictures during
  flight test or from the roof tops of buildings. This is a way to supplement the data
  we already have of real images.

- Improve synthetic data generation by adding pixel level variations in color, blurs,
  and other realistic artifacts. A useful task is comparing some synthetic data to real
  life data and analyzing the differences between the two.

- Research methods for using Generative Artifical Nets to augment synthetic data to look
  more like real data. This may or may not be feasible, but if successful, a low-cost
  way to generate better data.
