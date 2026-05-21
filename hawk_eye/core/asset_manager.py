"""Functions to pull in data from the cloud.

This relies on Google Cloud Storage python API. Please see the Image Recognition lead
to recieve the proper credentials for access to the bucket.

If you do not have Google Cloud APIs installed please run:

.. code-block:: bash

    $ hawk_eye/setup/install_google_cloud.sh
"""

import pathlib
import subprocess
import tarfile
import tempfile
from typing import List, Union

from google.cloud import storage

from hawk_eye.data_generation import generate_config as config

BUCKET = "uav_austin"


def pull_all() -> None:
    """Pull the backgrounds, base shapes and font files as specified in
    ``hawk_eye.data_generation.generate_config.py``."""
    pull_backgrounds()
    pull_base_shapes()
    pull_fonts()


def pull_backgrounds() -> None:
    """Pull the shape generation backgrounds.

    This function utilizes the ``BACKGROUNDS_URLS`` in
    ``hawk_eye.data_generation.generate_config.py``."""
    for background_archive in config.BACKGROUNDS_URLS:
        download_file(background_archive, config.ASSETS_DIR)


def pull_base_shapes() -> None:
    """Pull the base shapes.

    This function utilizes the ``BASE_SHAPES_URL`` in
    ``hawk_eye.data_generation.generate_config.py``."""
    for bash_shape_archive in config.BASE_SHAPES_URL:
        download_file(bash_shape_archive, config.ASSETS_DIR)


def pull_fonts() -> None:
    """Pull the fonts used in data generation.

    This function utilizes the ``FONTS_URL`` in
    ``hawk_eye.data_generation.generate_config.py``."""
    download_file(config.FONTS_URL, config.ASSETS_DIR)


def upload_file(source_path: pathlib.Path, destination: str):
    """A generic function for uploading a file from local storage to the cloud.

    Args:
        source_path: a path to the local file
        desination: where to upload the file

    :raises FileNotFoundError: if the object does not exist locally.
    """

    if not source_path.is_file():
        raise FileNotFoundError(f"Can't find {source_path}")

    bucket = _get_client_bucket()
    blob = bucket.blob(destination)
    blob.upload_from_filename(str(source_path))


def download_file(filename: str, destination: pathlib.Path) -> None:
    """Download an object from Google Cloud Storage.

    This function assumes the file is in ``.tar.gz`` format.

    Args:
        filename: the path to the file on Google Cloud. This should not include
            the bucket name.
        destination: the path to extract the tarball.

    :raises FileNotFoundError: if the object does not exist in Google Cloud.
    """
    filename = pathlib.Path(filename)
    folder_name = filename.stem.split(".", 1)[0]
    if not (destination / folder_name).is_dir():
        print(f"Fetching {filename}...", end="", flush=True)

        with tempfile.TemporaryDirectory() as d:
            tmp_file = pathlib.Path(d) / "file.tar.gz"
            client = storage.Client()
            bucket = client.get_bucket(BUCKET)
            blob = bucket.blob(str(filename))
            if not blob.exists():
                raise FileNotFoundError(f"Could not find {filename} on Google Cloud.")
            blob.download_to_filename(tmp_file)
            untar_and_move(tmp_file, destination)

            print(" done.")


def untar_and_move(filename: pathlib.Path, destination: pathlib.Path) -> None:
    """Extract tarball to directory.

    Args:
        filename: path to the archive to be extraced.
        destination: where to extract the archive.
    """

    print(f"Extracting to {destination}...", end="", flush=True)
    with tarfile.open(filename, "r") as tar:
        tar.extractall(destination)

    # Remove hidden files that might have been left behind by
    # the untarring.
    for filename in destination.rglob("._*"):
        filename.unlink()


def download_model(model_type: str, timestamp: str) -> pathlib.Path:
    """Download or reuse a local production model archive.

    Args:
        model_type: Model family to download, either ``classifier`` or ``detector``.
        timestamp: Timestamped model version to retrieve.

    Returns:
        Local directory containing the extracted model files.
    """
    assert model_type in ["classifier", "detector"], f"Unsupported model {model_type}."
    filename = f"{model_type}/{timestamp}.tar.gz"
    destination = pathlib.Path(f"~/runs/uav-{model_type}").expanduser() / timestamp

    if not destination.is_dir():
        download_file(filename, destination)

    return destination


def upload_model(model_type: str, path: pathlib.Path) -> None:
    """Upload a packaged model archive.

    Args:
        model_type: Model family to upload, either ``classifier`` or ``detector``.
        path: Local archive path to upload.
    """
    upload_file(path, f"{model_type}/{path.name}")


def _get_client_bucket():
    client = storage.Client()
    return client.get_bucket(BUCKET)
