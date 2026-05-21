#!/usr/bin/env python3
""" Collection of unittests to test data generation scripts. """

import pathlib
import tempfile
import unittest

from test import generate_config_test
from hawk_eye.data_generation import create_clf_data
from hawk_eye.data_generation import create_detection_data


class DetectionDataGeneration(unittest.TestCase):
    create_detection_data.config = generate_config_test

    def test_create_train_data(self) -> None:
        """Check that detector training data generation writes expected images."""
        num_imgs = generate_config_test.DET_TRAIN_IMAGES
        with tempfile.TemporaryDirectory() as d:
            tmp_dir = pathlib.Path(d)
            create_detection_data.generate_all_images(tmp_dir / "train", num_imgs, 0)
            self.assertEqual(len(list((tmp_dir / "train").rglob("*.png"))), num_imgs)

    def test_create_val_data(self) -> None:
        """Check that detector validation data generation writes expected images."""
        num_imgs = generate_config_test.DET_VAL_IMAGES
        with tempfile.TemporaryDirectory() as d:
            tmp_dir = pathlib.Path(d)
            create_detection_data.generate_all_images(tmp_dir / "val", num_imgs, 0)
            self.assertEqual(len(list((tmp_dir / "val").rglob("*.png"))), num_imgs)


class ClassificationDataGeneration(unittest.TestCase):
    create_clf_data.config = generate_config_test

    def test_create_data(self) -> None:
        """Check that classifier data generation writes expected images."""
        num_imgs = generate_config_test.CLF_IMAGES
        with tempfile.TemporaryDirectory() as d:
            tmp_dir = pathlib.Path(d)
            create_clf_data.create_clf_images(num_imgs, tmp_dir, val_fraction=0.0)
            self.assertEqual(
                len(list((tmp_dir / "clf_train").rglob("*.png"))), num_imgs
            )


if __name__ == "__main__":
    unittest.main()
