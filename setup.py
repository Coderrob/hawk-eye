#!/usr/bin/env python3

import distutils.command.build as build
import distutils.command.clean as clean
import pathlib
import platform
import setuptools
import shutil
import subprocess
import sys
import tempfile
from typing import List
import os

from setuptools.command import test

from hawk_eye.inference import production_models

__version__ = pathlib.Path("version.txt").read_text()

_MODELS_DIR = pathlib.Path.cwd() / "hawk_eye/core/production_models"


def _get_packages() -> List[str]:
    all_deps = pathlib.Path("hawk_eye/setup/requirements.txt").read_text().splitlines()
    deps = []
    deps_to_export = ["numpy", "pillow", "pyyaml"]
    for dep in all_deps:
        for export_dep in deps_to_export:
            if export_dep in dep:
                deps.append(dep)

    return deps


class Build(build.build):
    def initialize_options(self):
        build.build.initialize_options(self)

    def finalize_options(self):
        build.build.finalize_options(self)

    def run(self) -> None:
        self.run_command("prepare_models")
        build.build.run(self)


class Test(test.test):
    def run(self):
        self.run_command("prepare_models")
        test.test.run(self)


class PrepareModels(build.build):
    def run(self):
        _MODELS_DIR.mkdir()
        models = ["classification_model", "detection_model"]
        for model in models:
            self._prepare_model(model)

    def _prepare_model(self, model_target: str):

        bazel_command = ["bazel", "fetch", f"@{model_target}//..."]
        if subprocess.call(bazel_command) != 0:
            sys.exit(-1)

        bazel_external = (
            pathlib.Path(
                subprocess.check_output(["bazel", "info", "output_base"])
                .decode("utf-8")
                .strip()
            )
            / "external"
        )

        if "classification" in model_target:
            shutil.copytree(
                bazel_external
                / model_target
                / production_models._CLASSIFIER["timestamp"],
                _MODELS_DIR / production_models._CLASSIFIER["timestamp"],
            )
        else:
            shutil.copytree(
                bazel_external
                / model_target
                / production_models._DETECTOR["timestamp"],
                _MODELS_DIR / production_models._DETECTOR["timestamp"],
            )


try:
    setuptools.setup(
        name="hawk_eye",
        version=__version__,
        description=("Find targets"),
        author="UAV Austin Image Recognition",
        packages=setuptools.find_packages(),
        python_requires=">=3.11",
        cmdclass={"build": Build, "prepare_models": PrepareModels, "test": Test},
        include_package_data=True,
        install_requires=_get_packages(),
        test_suite="hawk_eye.inference",
    )
finally:
    shutil.rmtree(_MODELS_DIR)
