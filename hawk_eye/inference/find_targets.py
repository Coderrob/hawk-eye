#!/usr/bin/env python3
"""Contains logic for finding targets in images."""

import argparse
import pathlib
import json
from typing import List, Tuple

from PIL import Image
from PIL import ImageDraw
import numpy as np
import torch

from hawk_eye.core import classifier
from hawk_eye.core import detector
from hawk_eye.data_generation import generate_config as config
from hawk_eye import image_utils
from hawk_eye.inference import inference_types
from hawk_eye.inference import production_models
from third_party.detectron2 import postprocess


# Taken directly from albumentation src: augmentations/functional.py#L131.
# This is the only function we really need from albumentations for inference,
# so it is not worth requiring that as a dependency for distribution.
def normalize(
    img: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    max_pixel_value: float = 255.0,
) -> np.ndarray:
    """Normalize images based on ImageNet values. This is identical to
    albumentations normalization used for training.
    Args:
        img: The image to normalize.
        mean: The mean of each channel.
        std: The standard deviation of each channel.
        max_pixel_value: The max value a pixel can have.

    Returns:
        A normalized image as a numpy array.
    """

    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator

    return img


def _crop_tile(
    image: Image.Image, x: int, y: int, tile_size: Tuple[int, int]
) -> torch.Tensor:
    cropped = image.crop((x, y, x + tile_size[0], y + tile_size[1]))
    return torch.Tensor(normalize(np.array(cropped)))


def tile_image(
    image: Image.Image, tile_size: Tuple[int, int], overlap: int
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Take in an image and tile it into smaller tiles for inference.

    Args:
        image: The input image to tile.
        tile_size: The (width, height) of the tiles.
        overlap: The overlap between adjacent tiles (height, width).

    Returns:
        A tensor of the tiles and a list of the (x, y) offset for the tiles.
        The offets are needed to keep track of which tiles have targets.

    Examples::

        >>> tiles, coords = tile_image(Image.new("RGB", (1000, 1000)), (512, 512), 50)
        >>> tiles.shape[0]
        9
        >>> len(coords)
        9
        >>> tiles.shape[-2:]
        torch.Size([512, 512])
    """
    tiles, coords = [], []
    width, height = image.size

    for x, y in image_utils.tile_origins(width, height, tile_size, overlap):
        tiles.append(_crop_tile(image, x, y, tile_size))
        coords.append((x, y))

    # Transpose the images from BHWC -> BCHW
    tiles = torch.stack(tiles).permute(0, 3, 1, 2)

    return tiles, coords


def create_batches(
    image_tensor: torch.Tensor, coords: List[Tuple[int, int]], batch_size: int
):
    """Creates batches of images based on the supplied params. The whole image
    is tiled first, the batches are generated.

    Args:
        image: The opencv opened image.
        tile_size: The height, width of the tiles to create.
        overlap: The amount of overlap between tiles.
        batch_size: The number of images to have per batch.

    Returns:
        Yields the image batch and the top left coordinate of the tile in the
        space of the original image.
    """

    for idx in range(0, image_tensor.shape[0], batch_size):
        yield image_tensor[idx : idx + batch_size], coords[idx : idx + batch_size]


def _to_cuda_half(tiles: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        return tiles.cuda().half()
    return tiles


def load_models(
    clf_timestamp: str = production_models._CLASSIFIER["timestamp"],
    det_timestamp: str = production_models._DETECTOR["timestamp"],
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """ Loads the given time stamps for the classification and detector models.

    Args:
        clf_timestamp: Which classification model to load.
        det_timestamp: Which detection model to load.

    Returns:
        Returns both models.
    """

    clf_model = classifier.Classifier(
        timestamp=clf_timestamp, half_precision=torch.cuda.is_available()
    )
    clf_model.eval()

    # TODO(alex): Pass in the confidence for the detector.
    det_model = detector.Detector(
        timestamp=det_timestamp,
        confidence=0.2,
        half_precision=torch.cuda.is_available(),
    )
    det_model.eval()

    # Do FP16 when inferencing
    if torch.cuda.is_available():
        det_model.cuda()
        det_model.half()

        clf_model.cuda()
        clf_model.half()

    return clf_model, det_model


def find_all_targets(
    images: List[pathlib.Path],
    clf_timestamp: str = production_models._CLASSIFIER["timestamp"],
    det_timestamp: str = production_models._DETECTOR["timestamp"],
    visualization_dir: pathlib.Path = None,
    save_json_data: bool = False,
) -> None:
    """ Entrypoint function if running this script as main.

    Args:
        images: A list of all the images to inference.
        clf_timestamp: The classification model to load.
        det_timestamp: The detection model to load.
        visualization_dir: Where to save the visualizations, if any.
    """

    clf_model, det_model = load_models(clf_timestamp, det_timestamp)

    for image_path in images:
        _process_image(
            image_path, clf_model, det_model, visualization_dir, save_json_data
        )


def _process_image(
    image_path: pathlib.Path,
    clf_model: torch.nn.Module,
    det_model: torch.nn.Module,
    visualization_dir: pathlib.Path,
    save_json_data: bool,
) -> None:
    image = Image.open(image_path)
    assert image is not None, f"Could not read {image_path}."

    targets, target_tiles = find_targets(image, clf_model, det_model)
    _save_outputs(
        image_path, image, visualization_dir, targets, target_tiles, save_json_data
    )


def _save_outputs(
    image_path: pathlib.Path,
    image: Image.Image,
    visualization_dir: pathlib.Path,
    targets: List[inference_types.Target],
    target_tiles: List[Tuple[int, int]],
    save_json_data: bool,
) -> None:
    if visualization_dir is not None:
        visualize_image(
            image_path.name,
            np.array(image),
            visualization_dir,
            targets,
            target_tiles,
        )
    if save_json_data:
        save_target_meta(
            (visualization_dir / image_path.name).with_suffix(".json"),
            image_path.name,
            targets,
        )


def _classify_tiles(
    clf_model: torch.nn.Module, tiles_batch: torch.Tensor, clf_confidence: float
) -> torch.Tensor:
    tiles = torch.nn.functional.interpolate(tiles_batch, config.PRECLF_SIZE)
    return clf_model.classify(tiles, probability=True)[:, 1] >= clf_confidence


def _positive_coords(
    coords: List[Tuple[int, int]], predictions: torch.Tensor
) -> List[Tuple[int, int]]:
    pred_values = predictions.detach().cpu().tolist()
    return [coords[idx] for idx, value in enumerate(pred_values) if value]


def _detect_tiles(
    det_model: torch.nn.Module,
    tiles_batch: torch.Tensor,
    coords: List[Tuple[int, int]],
) -> List[Tuple[Tuple[int, int], List[postprocess.BoundingBox]]]:
    detections = []
    for det_tiles, det_coords in create_batches(tiles_batch, coords, 15):
        det_tiles = torch.nn.functional.interpolate(det_tiles, config.DETECTOR_SIZE)
        detections.extend(zip(det_coords, det_model(det_tiles)))
    return detections


@torch.no_grad()
def find_targets(
    image: Image.Image,
    clf_model: torch.nn.Module,
    det_model: torch.nn.Module,
    clf_confidence: float = 0.9,
) -> Tuple[List[inference_types.Target], List[Tuple[int, int]]]:
    """ Tile up image, classify them, then perform object detection where it's needed.

    Args:
        image: The input image to inference.
        clf_model: The loaded classification model.
        det_model: The loaded detection model.

    Returns:
        Detected targets in original image coordinates and tile coordinates that
        passed the classifier.
    """

    image_tensor, coords = tile_image(image, config.CROP_SIZE, config.CROP_OVERLAP)
    target_tiles, retval = [], []

    for tiles_batch, coords in create_batches(image_tensor, coords, 200):
        tiles_batch = _to_cuda_half(tiles_batch)
        preds = _classify_tiles(clf_model, tiles_batch, clf_confidence)
        positive_coords = _positive_coords(coords, preds)
        target_tiles.extend(positive_coords)
        retval.extend(_detect_tiles(det_model, tiles_batch[preds], positive_coords))

    return globalize_boxes(retval, config.CROP_SIZE[0]), target_tiles


def globalize_boxes(
    results: List[postprocess.BoundingBox], img_size: int
) -> List[inference_types.Target]:
    """Take the normalized detections on a _tile_ and gloabalize them to pixel space of
    the original large image.

    Args:
        results: A list of the detections for the tiles.
        img_size: The size of the tile whihc is needed to unnormalize the detections.

    Returns:
        A list of the globalized boxes
    """

    final_targets = []
    img_size = torch.Tensor([img_size] * 4)

    for coords, bboxes in results:
        for box in bboxes:
            relative_coords = box.box * img_size
            relative_coords += torch.Tensor(2 * list(coords)).int()
            final_targets.append(
                inference_types.Target(
                    x=int(relative_coords[0]),
                    y=int(relative_coords[1]),
                    width=int(relative_coords[2] - relative_coords[0]),
                    height=int(relative_coords[3] - relative_coords[1]),
                    shape=inference_types.Shape[
                        config.OD_CLASSES[box.class_id].upper().replace("-", "_")
                    ],
                )
            )

    return final_targets


def visualize_image(
    image_name: str,
    image: np.ndarray,
    visualization_dir: pathlib.Path,
    targets: List[inference_types.Target],
    clf_tiles: List[Tuple[int, int]],
) -> None:
    """Function used to draw boxes and information onto image for visualizing the output
    of inference.

    Args:
        image_name: The original image name used for saving the visualization.
        image: The image array.
        visualization_dir: Where to save the visualizations.
        targets: A list of the targets that were found during inference.
        clf_tiles: Which tiles were classified as having targets in them.
    """
    # Create a PIL drawable object.
    image = Image.fromarray(image).convert("RGBA")
    image_draw = ImageDraw.Draw(image)

    # Draw the target rectangles.
    for target in targets:
        top_left = (target.x, target.y)
        bottom_right = (target.x + target.width, target.y + target.height)
        image_draw.rectangle([top_left, bottom_right], outline=(0, 255, 0), width=2)
        image_draw.text(
            (target.x, target.y - 10), target.shape.name.lower(), (0, 255, 0)
        )

    # Draw an overlay onto the tiles which were classified as having targets.
    w, h = config.CROP_SIZE
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for group in clf_tiles:
        x, y = group
        draw.rectangle([(x, y), (x + w, y + h)], fill=(0, 0, 0, 40))
        image = Image.alpha_composite(image, overlay)

    image.convert("RGB").save((visualization_dir / image_name))


# TODO(alex) use this for writing jsons.
def save_target_meta(
    filename_meta: pathlib.Path,
    filename_image: str,
    targets: List[inference_types.Target],
) -> None:
    """Save target metadata to a file.

    Args:
        filename_meta: Output JSON metadata path.
        filename_image: Original image filename associated with the targets.
        targets: Targets to serialize.
    """
    meta = {}
    for idx, target in enumerate(targets):
        meta[f"target-{idx}"] = {
            "x": target.x,
            "y": target.y,
            "width": target.width,
            "height": target.height,
            "orientation": target.orientation,
            "shape": target.shape.name.lower(),
            "background_color": target.background_color.name.lower(),
            "alphanumeric": target.alphanumeric,
            "alphanumeric_color": target.alphanumeric_color.name.lower(),
            "image": filename_image,
            "confidence": target.confidence,
        }
    filename_meta.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script used to find targets in an image"
    )
    parser.add_argument(
        "--image_path",
        required=False,
        type=pathlib.Path,
        help="Path to an image to inference.",
    )
    parser.add_argument(
        "--image_dir",
        required=False,
        type=pathlib.Path,
        help="Path to directory of images to inference.",
    )
    parser.add_argument(
        "--image_extension",
        required=False,
        type=str,
        help="Needed when an image directory is supplied.",
    )
    parser.add_argument(
        "--clf_timestamp",
        required=False,
        type=str,
        default=production_models._CLASSIFIER["timestamp"],
        help="Timestamp of the classifier model to use.",
    )
    parser.add_argument(
        "--det_timestamp",
        required=False,
        type=str,
        default=production_models._DETECTOR["timestamp"],
        help="Timestamp of the detector model to use.",
    )
    parser.add_argument(
        "--visualization_dir",
        required=False,
        type=pathlib.Path,
        help="Optional directory to save visualization to.",
    )
    parser.add_argument(
        "--save_json_data", action="store_true", help="Save JSON data if specified.",
    )
    args = parser.parse_args()

    # Get either the image or images
    if args.image_path is None and args.image_dir is None:
        raise ValueError("Please supply either an image or directory of images.")
    if args.image_path is not None:
        imgs = [args.image_path.expanduser()]
    elif args.image_dir is not None:
        assert args.image_extension.startswith(".")
        imgs = args.image_dir.expanduser().glob(f"*{args.image_extension}")

    viz_dir = None
    if args.visualization_dir is not None:
        viz_dir = args.visualization_dir.expanduser()
        viz_dir.mkdir(exist_ok=True, parents=True)

    find_all_targets(
        imgs, args.clf_timestamp, args.det_timestamp, viz_dir, args.save_json_data
    )
