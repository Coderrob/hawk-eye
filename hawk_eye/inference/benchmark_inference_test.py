#!/usr/bin/env python3

"""Simple test to ensure inference benchmarking function works."""

import unittest

from hawk_eye.inference import benchmark_inference
from hawk_eye.inference import production_models


class BenchmarkInference(unittest.TestCase):
    def test_classifier(self) -> None:
        """Check that classifier benchmarking runs."""
        benchmark_inference.benchmark(
            timestamp=production_models.PROD_MODELS["classifier"]["timestamp"],
            batch_size=1,
            model_type="classifier",
            run_time=2.5,
        )

    def test_detector(self) -> None:
        """Check that detector benchmarking runs."""
        benchmark_inference.benchmark(
            timestamp=production_models.PROD_MODELS["detector"]["timestamp"],
            batch_size=1,
            model_type="detector",
            run_time=2.5,
        )


if __name__ == "__main__":
    unittest.main()
