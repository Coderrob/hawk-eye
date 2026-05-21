#!/usr/bin/env python3
"""  This script creates classification data for the pre-detection classification step;
however, this script relies upon detection already existing. Also, we need to ensure our
output dataset does not have repeats of image slices between the train and validation
sets -- this would mess up our training metrics. Since the detector data can contain
empty tiles, we will not be copying those here. Instead, we prefer to generate our own
empty slices so we have precise cnotrol over which exist. We will first generate all the
background crops and copy the target crops into one folder, then we'll shuffle them all
and split the data into 80% training and 20% validation. """

import json
import pathlib
import random
import shutil
import tempfile

from PIL import Image
import tqdm

from hawk_eye.core import asset_manager
from hawk_eye.data_generation import generate_config as config
from hawk_eye.data_generation import create_detection_data

# Get constants from config
CLF_WIDTH, CLF_HEIGHT = config.PRECLF_SIZE
CROP_WIDTH, CROP_HEIGHT = config.CROP_SIZE


def create_clf_images(
    num_gen: int,
    save_dir: pathlib.Path = config.DATA_DIR,
    val_fraction: float = config.CLF_VAL_FRACTION,
) -> None:
    """Generate data for the classifier model.

    Args:
        num_gen: Number of classifier images to generate.
        save_dir: Directory where classifier data should be written.
        val_fraction: Fraction of data to reserve for validation.
    """

    val_int = int(val_fraction * 100)
    _ensure_detector_data(save_dir)

    with tempfile.TemporaryDirectory() as d:
        tmp_dir = pathlib.Path(d)
        print("Copying target tiles.")
        _copy_target_tiles(_detector_images(save_dir), tmp_dir, num_gen)
        _create_background_tiles(tmp_dir, num_gen)
        _split_classifier_data(tmp_dir, save_dir, val_int)


def _ensure_detector_data(save_dir: pathlib.Path) -> None:
    if not (config.DATA_DIR / "detector_train").is_dir():
        create_detection_data.generate_all_images(
            save_dir / "detector_train", config.DET_TRAIN_IMAGES
        )
    if not (config.DATA_DIR / "detector_val").is_dir():
        create_detection_data.generate_all_images(
            save_dir / "detector_val", config.DET_VAL_IMAGES
        )


def _detector_images(save_dir: pathlib.Path):
    imgs = []
    imgs.extend(list((save_dir / "detector_train").rglob(f"*{config.IMAGE_EXT}")))
    imgs.extend(list((save_dir / "detector_val").rglob(f"*{config.IMAGE_EXT}")))
    random.shuffle(imgs)
    return imgs


def _copy_target_tiles(imgs, tmp_dir: pathlib.Path, num_gen: int) -> None:
    idx = 0
    for image_path in tqdm.tqdm(_target_images(imgs)):
        image = Image.open(image_path).resize(config.PRECLF_SIZE)
        image.save(tmp_dir / f"target_{idx}{image_path.suffix}")
        idx += 1
        if idx == num_gen:
            break


def _target_images(imgs):
    for image_path in imgs:
        if _has_bboxes(image_path):
            yield image_path


def _has_bboxes(image_path: pathlib.Path) -> bool:
    return bool(json.loads(image_path.with_suffix(".json").read_text())["bboxes"])


def _create_background_tiles(tmp_dir: pathlib.Path, num_gen: int) -> None:
    num_tiles = 0
    for idx, img in tqdm.tqdm(enumerate(create_detection_data.get_backgrounds())):
        num_tiles = single_clf_image(img, idx, num_gen, tmp_dir, num_tiles)


def _split_classifier_data(
    tmp_dir: pathlib.Path, save_dir: pathlib.Path, val_int: int
) -> None:
    train_dir = save_dir / "clf_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir = save_dir / "clf_val"
    val_dir.mkdir(parents=True, exist_ok=True)

    imgs = list(tmp_dir.glob("*"))
    random.shuffle(imgs)
    for img in imgs:
        target_dir = val_dir if random.randint(0, 100) < val_int else train_dir
        shutil.move(img, target_dir / img.name)


def single_clf_image(
    image: Image.Image,
    number: int,
    num_gen: int,
    save_dir: pathlib.Path,
    num_tiles: int,
) -> int:
    """Slice out crops from the original background image and save to disk. NOTE: we do
    not have any overlap between adjacent tiles because we want to avoid having any
    leakage between images. With data leakage, we might end up with two adjacent tiles in
    both the train and eval set.

    Args:
        image: Background image path.
        number: Background image index.
        num_gen: Maximum number of background tiles to generate.
        save_dir: Directory where generated tiles should be saved.
        num_tiles: Number of tiles already generated.

    Returns:
        Updated number of generated background tiles.
    """
    image = Image.open(image)
    tile_num = 0
    for x, y in _background_origins(image):
        if num_tiles > num_gen:
            break
        crop = image.crop((x, y, x + config.CROP_SIZE[0], y + config.CROP_SIZE[1]))
        crop = crop.resize(config.PRECLF_SIZE)
        save_path = (
            save_dir / f"background_{number}_{tile_num}_{x}_{y}{config.IMAGE_EXT}"
        )
        crop.save(save_path)
        num_tiles += 1
        tile_num += 1

    return num_tiles


def _background_origins(image: Image.Image):
    for x in range(0, image.size[0] - config.CROP_SIZE[1], config.CROP_SIZE[0]):
        for y in range(0, image.size[1] - config.CROP_SIZE[1], config.CROP_SIZE[1]):
            yield x, y


if __name__ == "__main__":
    random.seed(42)

    if config.CLF_IMAGES != 0:
        asset_manager.pull_all()
        create_clf_images(config.CLF_IMAGES, config.DATA_DIR, config.CLF_VAL_FRACTION)
