#!/usr/bin/env python3
"""Testing of the asset_manager utilities to ensure some generally reliable
usage."""

import datetime
import pathlib
import tarfile
import tempfile
import unittest

from hawk_eye.core import asset_manager

# Upload the test files to this test bucket. Items in this bucket are deleted
# after 24 hrs.
asset_manager.BUCKET = "uav_austin_test"


class FileUpload(unittest.TestCase):
    def test_file_upload(self) -> None:
        """Check that an existing local file can be uploaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = pathlib.Path(tmp_dir)
            tmp_file = tmp_dir / f"{datetime.datetime.now().isoformat()}.txt"
            tmp_file.write_text("")
            asset_manager.upload_file(tmp_file, f"{tmp_file.name}")

            bucket = asset_manager._get_client_bucket()
            blob = bucket.blob(f"{tmp_file.name}")
            self.assertTrue(blob.exists())

    def test_no_file(self) -> None:
        """Check that uploading a missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            asset_manager.upload_file(
                pathlib.Path(tempfile.NamedTemporaryFile().name), ""
            )


class FileDownload(unittest.TestCase):
    def test_download_file(self):
        """Check that an uploaded tarball can be downloaded and extracted."""

        # Create and upload the file
        with tempfile.TemporaryDirectory() as d:
            tmp_dir = pathlib.Path(d)
            tmp_file = tmp_dir / "file.txt"
            tmp_file.touch()

            archive_name = tmp_dir / f"{datetime.datetime.now().isoformat()}.tar.gz"
            with tarfile.open(archive_name, mode="w:gz") as tar:
                for tmp_file in tmp_dir.glob("*"):
                    tar.add(tmp_file)

            asset_manager.upload_file(archive_name, f"{archive_name.name}")

        # Download and extract
        with tempfile.TemporaryDirectory() as d:
            asset_manager.download_file(f"{archive_name.name}", tmp_dir / "dir")

        self.assertTrue((tmp_dir / "dir").is_dir())

    def test_no_remote_file(self) -> None:
        """Check that downloading a missing remote file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            with tempfile.TemporaryDirectory() as d:
                asset_manager.download_file(
                    f"{datetime.datetime.now().isoformat()}.tar.gz", pathlib.Path(d),
                )


if __name__ == "__main__":
    unittest.main()
