#!/usr/bin/env python3
"""Contains code to slice up an image into smaller tiles."""

__author__ = "Kadhir Umasankar"

import argparse
import pathlib
from typing import List, Tuple

import numpy as np
from PIL import Image
import tqdm

from hawk_eye import image_utils
from hawk_eye.data_generation import generate_config as config


def slice_image(
    images: List, tile_size: Tuple[int, int], overlap: int, save_dir: pathlib.Path
) -> None:
    """ Take in an image and slice it into smaller images.
    Args:
        images: A list of paths to images to be sliced.
        tile_size: The (width, height) of the tiles.
        overlap: The overlap between adjacent tiles.
        save_dir: The path to the directory within which to save slices.
    Returns:
        None.
    """
    for filename in tqdm.tqdm(images, desc="Slicing images", total=len(images)):

        image = Image.open(filename)
        width, height = image.size

        for x, y in image_utils.tile_origins(width, height, tile_size, overlap):
            tile = image.crop((x, y, x + tile_size[0], y + tile_size[1]))
            tile.save(save_dir / f"{filename.stem}-{x}-{y}{filename.suffix}")

    (save_dir / "labels.txt").write_text("\n".join(config.SHAPE_TYPES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image_dir",
        type=pathlib.Path,
        required=False,
        help="Path to directory of images to be sliced.",
    )
    parser.add_argument(
        "--image_path",
        type=pathlib.Path,
        required=False,
        help="Path to an image to be sliced.",
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        required=False,
        default=".JPG,.jpg",
        help="A comma-separated list of extensions of images to slice.",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        required=True,
        help="Path to directory in which to store sliced images.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of pixels of overlap between adjacent slices.",
    )
    args = parser.parse_args()

    # Raises an error if none of image arguments are provided
    if args.image_path is None and args.image_dir is None:
        raise ValueError("Please supply either an image or directory of images.")

    # Extracts extensions from the input argument
    exts = [
        f".{ext}" if "." not in ext else ext for ext in args.image_extensions.split(",")
    ]

    if args.image_path is not None:
        if args.image_path.suffix in exts:
            images = [args.image_path.expanduser()]

    elif args.image_dir is not None:

        # If image_dir points to a directory, paths to all the images with the correct
        # extension are put in a list
        image_dir = args.image_dir.expanduser()

        if image_dir.is_dir():
            images = []

            for ext in exts:
                images.extend(list(image_dir.glob(f"*{ext}")))
        else:
            raise ValueError("Please supply a valid path to a directory.")

    # Makes a directory to save slices in
    save_dir = args.save_dir.expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)

    slice_image(
        images=images,
        tile_size=config.CROP_SIZE,
        overlap=args.overlap,
        save_dir=save_dir,
    )
