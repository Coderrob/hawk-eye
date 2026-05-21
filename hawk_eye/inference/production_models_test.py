# /usr/bin/env python3

"""Very simple check to ensure the production models listed have correct attributes."""

import unittest

from google.cloud import storage

from hawk_eye.core import asset_manager
from hawk_eye.inference import production_models


class ProductionModels(unittest.TestCase):

    _model_types = ["classifier", "detector"]
    _client = storage.Client()
    _bucket = _client.get_bucket(asset_manager.BUCKET)

    def _check_attributes(self, model: dict, model_type: str) -> bool:
        """Check that the model dictionary has a proper model type and that the
        sha256sum exists on GCloud."""

        file_path = f"{model_type}/{model['timestamp']}.tar.gz"

        return self._bucket.blob(str(file_path)).exists()

    def test_classifier(self) -> None:
        """Check that the production classifier metadata points to GCS."""

        self.assertTrue("classifier" in production_models.PROD_MODELS)
        self.assertTrue("sha256" in production_models._CLASSIFIER)
        self.assertTrue(
            self._check_attributes(production_models._CLASSIFIER, "classifier")
        )

    def test_detector(self) -> None:
        """Check that the production detector metadata points to GCS."""

        self.assertTrue("detector" in production_models.PROD_MODELS)
        self.assertTrue("sha256" in production_models._DETECTOR)
        self.assertTrue(self._check_attributes(production_models._DETECTOR, "detector"))


if __name__ == "__main__":
    unittest.main()
