.. hawk-eye documentation master file, created by
   sphinx-quickstart on Wed Jan 13 19:35:43 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

hawk-eye
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


An Application to detect Ground-Based targets.

Structure
~~~~~~~~~~~~~~~~~~~~~~
- **configs** - different configuration files for model training.
- **core** - a shared library accross different applications.
- **data_generation** - scripts for genertion datasets.
- **inference** - code for performing inference.
- **train** - scripts related to training models.


.. toctree::
    :caption: Setup
    :maxdepth: 2

    setup/setup


.. toctree::
    :caption: Core

    core/asset_manager
    core/classifier
    core/detector


.. toctree::
    :caption: Data Generation

    data_generation/overview.rst
    data_generation/create_clf_data
    data_generation/create_detection_data
    data_generation/real_data.rst


.. toctree::
    :caption: Training

    training


.. toctree::
    :caption: Inference

    inference/overview
    inference/benchmark_inference
    inference/find_targets
    inference/inference_types
    inference/production_models

.. toctree::
    :caption: Distribution

    distribution

.. toctree::
    :caption: Google Cloud Storage

    google_cloud

.. toctree::
    :caption: Testing

    tests

.. toctree::
    :caption: Maintenance

    architecture
    quality_gates


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
