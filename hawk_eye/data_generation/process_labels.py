#!/usr/bin/env python3
"""A script to parse the labels returned from Make Sense labeling jobs."""

import argparse
import csv
import pathlib
import json
import shutil
import tarfile
import tempfile

from hawk_eye.core import asset_manager
from hawk_eye.data_generation import generate_config
from hawk_eye.data_generation import create_detection_data

_GCS_DATASET_FOLDER = "real-target-datasets"


def parse_labels(
    image_dir: pathlib.Path,
    save_dir: pathlib.Path,
    csv_path: pathlib.Path,
    val_percent: int,
    upload: bool,
) -> None:
    """Entrypoint function for the script.

    Args:
        image_dir: path to the tiles that were uploaded for labeling.
        save_dir: where to save the dataset.
        csv_path: path to the labels csv downloaded from Make Sense.
        val_percent: an int specifying the percentage of data to use for
            validation.
        upload: Whether or not to upload the dataset.
    """
    save_dir = save_dir / "images"
    save_dir.mkdir(exist_ok=True, parents=True)
    images = _read_image_names(csv_path)
    _write_tile_labels(image_dir, save_dir, csv_path, images)
    _create_splits(image_dir, save_dir, images, val_percent)
    _upload_dataset(save_dir, upload)


def _read_image_names(csv_path: pathlib.Path) -> list:
    with open(csv_path, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        images = []
        for row in spamreader:
            vals = row[0].split(",")
            images.append(vals[-3])
    return sorted(images)


def _write_tile_labels(
    image_dir: pathlib.Path,
    save_dir: pathlib.Path,
    csv_path: pathlib.Path,
    images: list,
) -> None:
    with open(csv_path, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")

        for row in spamreader:
            label, original_tile_path, tile_save_path = _tile_label(
                row[0].split(","), image_dir, save_dir, images
            )
            tile_save_path.with_suffix(".json").write_text(json.dumps(label, indent=2))
            shutil.copy2(original_tile_path, tile_save_path)


def _tile_label(
    vals: list, image_dir: pathlib.Path, save_dir: pathlib.Path, images: list
) -> tuple:
    original_tile_path = image_dir / vals[-3]
    class_name = vals[0]
    x1, y1, w, h = vals[1:5]
    img_w, img_h = vals[-2], vals[-1]
    label = {
        "bboxes": [
            {
                "class_id": generate_config.SHAPE_TYPES.index(class_name),
                "x1": float(x1) / float(img_w),
                "y1": float(y1) / float(img_h),
                "w": float(w) / float(img_w),
                "h": float(h) / float(img_h),
            },
        ],
        "image_id": images.index(original_tile_path.name),
    }
    return label, original_tile_path, save_dir / vals[-3]


def _create_splits(
    image_dir: pathlib.Path, save_dir: pathlib.Path, images: list, val_percent: int
) -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp_train, tmp_val = _create_split_dirs(pathlib.Path(d))
        val_num = int(len(images) * val_percent / 100)
        val_imgs = images[:val_num]
        train_imgs = images[val_num:]

        _copy_split(save_dir, tmp_val, val_imgs)
        _copy_split(image_dir, tmp_train, train_imgs, labels_dir=save_dir)

        if val_percent < 100:
            create_detection_data.create_coco_metadata(
                tmp_train, save_dir.parent / "train_coco.json"
            )
        create_detection_data.create_coco_metadata(
            tmp_val, save_dir.parent / "val_coco.json"
        )


def _create_split_dirs(tmp_dir: pathlib.Path) -> tuple:
    tmp_train = tmp_dir / "train"
    tmp_train.mkdir()
    tmp_val = tmp_dir / "val"
    tmp_val.mkdir()
    return tmp_train, tmp_val


def _copy_split(
    image_source: pathlib.Path,
    split_dir: pathlib.Path,
    images: list,
    labels_dir: pathlib.Path = None,
) -> None:
    labels_dir = labels_dir or image_source
    for img in images:
        shutil.copy2(image_source / img, split_dir / img)
        shutil.copy2(
            (labels_dir / img).with_suffix(".json"),
            (split_dir / img).with_suffix(".json"),
        )


def _upload_dataset(save_dir: pathlib.Path, upload: bool) -> None:
    if upload:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_archive = pathlib.Path(tmp_dir) / f"{save_dir.parent.name}.tar.gz"

            with tarfile.open(tmp_archive, "w:gz") as tar:
                tar.add(save_dir.parent, arcname=save_dir.parent.name)

            destination = f"{_GCS_DATASET_FOLDER}/{tmp_archive.name}"
            asset_manager.upload_file(tmp_archive, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image_dir",
        type=pathlib.Path,
        required=False,
        help="Path to directory of tiles that were labeled.",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        required=True,
        help="Path to directory in which to store sliced images.",
    )
    parser.add_argument(
        "--csv_path",
        type=pathlib.Path,
        required=True,
        help="Path to the csv outputted from labeling app.",
    )
    parser.add_argument(
        "--val_percent",
        type=int,
        help="Fraction of data to use for validation.",
        default=20,
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload the dataset to GCS.",
    )
    args = parser.parse_args()

    parse_labels(
        args.image_dir.expanduser(),
        args.save_dir.expanduser(),
        args.csv_path.expanduser(),
        args.val_percent,
        args.upload,
    )
