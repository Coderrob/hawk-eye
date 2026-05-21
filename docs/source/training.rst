Training
===========================

There are two types of training currently supported: classifier and detection.


Structure
---------------------------
Most machine learning model training pipelines can be broken down into a few
major components: the model, dataset, and loss functions.



Examples
---------------

Classification:
::

    PYTHONPATH=. hawk_eye/train/train_clf.py \
        --config hawk_eye/configs/vovnet.yaml

Detection:
::

    PYTHONPATH=. hawk_eye/train/train_det.py \
        --config hawk_eye/configs/vovnet-det.yaml

Fine Tuning
---------------
It is an unfortunate fact that most deep learning models need to be trained on
GPUs to be done in a reasonable amount of time.

If you only have a CPU, you might want to finetune a model. This means taking
an model intially trained on lots of data and training it on yours. This can
drastically speed up training.

.. automodule:: hawk_eye.train.classification
   :members:

.. automodule:: hawk_eye.train.detection
   :members:
