#!/usr/bin/env python3
""" This script generates training data for the object detector model. The output will be
images and the corresponding COCO metadata jsons. For most RetinaNet related training we
can train on images with _and without_ targets. Training on images without any targets
is valuable so the model sees that not every image will have a target, as this is the
real life case. """

import dataclasses
from typing import List, Tuple
import multiprocessing
import random
import json
import pathlib

from tqdm import tqdm
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont
from PIL import ImageOps

from hawk_eye.data_generation import generate_config as config
from hawk_eye.data_generation import image_ops
from hawk_eye.core import asset_manager


@dataclasses.dataclass
class alpha_params:
    font_multiplier: Tuple[int, int]


# Get constants from config
MAX_SHAPES = int(config.MAX_PER_SHAPE)
FULL_SIZE = config.FULL_SIZE
TARGET_COLORS = config.TARGET_COLORS
ALPHA_COLORS = config.ALPHA_COLORS
COLORS = config.COLORS
CLASSES = config.OD_CLASSES
ALPHAS = config.ALPHAS
_ALPHA_INFO = {
    "circle": alpha_params((0.35, 0.65)),
    "cross": alpha_params((0.35, 0.65)),
    "pentagon": alpha_params((0.35, 0.65)),
    "quarter-circle": alpha_params((0.35, 0.65)),
    "rectangle": alpha_params((0.35, 0.7)),
    "semicircle": alpha_params((0.35, 0.75)),
    "square": alpha_params((0.30, 0.8)),
    "star": alpha_params((0.25, 0.55)),
    "trapezoid": alpha_params((0.25, 0.75)),
    "triangle": alpha_params((0.25, 0.6)),
}
_ALPHA_OFFSETS = {
    "trapezoid": (0, -20),
    "triangle": (-24, 12),
    "quarter-circle": (14, -40),
    "cross": (0, -25),
    "square": (0, -10),
    "circle": (0, -50),
}


def generate_all_images(gen_type: pathlib.Path, num_gen: int, offset: int = 0) -> None:
    """Main function which prepares all the relevant information regardining data
    generation. Data will be generated using a multiprocessing pool for efficiency.

    Args:
        gen_type: The name of the data being generated.
        num_gen: The number of images to generate.
        offset: TODO(alex): Are we still using this?
    """
    # Make the proper folders for storing the data.
    images_dir = pathlib.Path(gen_type) / "images"
    images_dir.mkdir(exist_ok=True, parents=True)

    r_state = random.getstate()
    random.seed(f"{gen_type}{offset}")

    # All the random selection is generated ahead of time, that way
    # the process can be resumed without the shapes changing on each
    # run.

    base_shapes = {shape: get_base_shapes(shape) for shape in config.SHAPE_TYPES}

    numbers = list(range(offset, num_gen + offset))

    # Create a random list of the background files.
    backgrounds = random_list(get_backgrounds(), num_gen)
    flip_bg = random_list([False, True], num_gen)
    mirror_bg = random_list([False, True], num_gen)
    blurs = random_list([1], num_gen)
    num_targets = random_list(range(1, MAX_SHAPES), num_gen)

    crop_xs = random_list(range(0, config.FULL_SIZE[0] - config.CROP_SIZE[0]), num_gen)
    crop_ys = random_list(range(0, config.FULL_SIZE[1] - config.CROP_SIZE[1]), num_gen)

    num_targets = 1
    shape_params = []

    for i in range(num_gen):
        shape_names = random_list(config.SHAPE_TYPES, num_targets)
        bases = [random.choice(base_shapes[shape]) for shape in shape_names]
        alphas = random_list(config.ALPHAS, num_targets)
        font_files = random_list(config.ALPHA_FONTS, num_targets)

        target_colors = random_list(TARGET_COLORS, num_targets)
        alpha_colors = random_list(ALPHA_COLORS, num_targets)

        # Make sure shape and alpha are different colors
        for i, target_color in enumerate(target_colors):
            while alpha_colors[i] == target_color:
                alpha_colors[i] = random.choice(ALPHA_COLORS)

        target_rgbs = [random.choice(COLORS[color]) for color in target_colors]
        alpha_rgbs = [random.choice(COLORS[color]) for color in alpha_colors]

        sizes = random_list(range(20, 100), num_targets)

        angles = random_list(range(0, 360), num_targets)

        xs = random_list(range(120, config.CROP_SIZE[0] - 120), num_targets)
        ys = random_list(range(120, config.CROP_SIZE[1] - 120), num_targets)

        shape_params.append(
            list(
                zip(
                    shape_names,
                    bases,
                    alphas,
                    font_files,
                    sizes,
                    angles,
                    target_colors,
                    target_rgbs,
                    alpha_colors,
                    alpha_rgbs,
                    xs,
                    ys,
                )
            )
        )

    random.setstate(r_state)

    # Generate data in a multiprocessing pool to use the specified amount of CPU
    # resources. We have to be careful to ensure we do not access and background image
    # at the same time. We use a manager dictionary to let us know if the image to be
    # opened is already being read somewhere else.
    data = zip(
        numbers,
        backgrounds,
        crop_xs,
        crop_ys,
        flip_bg,
        mirror_bg,
        blurs,
        shape_params,
        [gen_type] * num_gen,
    )

    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(generate_single_example, data)

        for _ in tqdm(processes, total=num_gen):
            pass

    create_coco_metadata(
        gen_type / "images",
        gen_type / ("val_coco.json" if "val" in gen_type.name else "train_coco.json"),
    )


def generate_single_example(data) -> None:
    """Create one synthetic detector image and label file.

    Args:
        data: Packed generation parameters for a single image.
    """
    (
        number,
        background_path,
        crop_x,
        crop_y,
        flip_bg,
        mirror_bg,
        blur,
        shape_params,
        gen_type,
    ) = data

    data_path = config.DATA_DIR / gen_type / "images"
    labels_fn = data_path / f"ex_{number}.json"
    img_fn = data_path / f"ex_{number}{config.IMAGE_EXT}"

    background = PIL.Image.open(background_path)
    assert background is not None
    background = background.crop(
        (crop_x, crop_y, crop_x + config.CROP_SIZE[0], crop_y + config.CROP_SIZE[1])
    )

    if flip_bg:
        background = ImageOps.flip(background)
    if mirror_bg:
        background = ImageOps.mirror(background)

    if random.randint(0, 100) / 100 >= config.EMPTY_TILE_PROB:
        shape_imgs = [create_shape(*shape_param) for shape_param in shape_params]
        shape_bboxes, background = add_shapes(
            background, shape_imgs, shape_params, blur
        )
    else:
        shape_bboxes = []

    background = background.resize(config.DETECTOR_SIZE)
    background.save(img_fn)

    objects = [
        {
            "class_id": shape_bbox[0],
            "x1": shape_bbox[1],
            "y1": shape_bbox[2],
            "w": shape_bbox[3],
            "h": shape_bbox[4],
        }
        for shape_bbox in shape_bboxes
    ]

    labels_fn.write_text(json.dumps({"bboxes": objects, "image_id": number}, indent=2))


def add_shapes(
    background: PIL.Image.Image,
    shape_imgs: PIL.Image.Image,
    shape_params,
    blur_radius: int,
) -> Tuple[List[Tuple[int, int, int, int, int]], PIL.Image.Image]:
    """Paste shapes onto a background image.

    Args:
        background: Background crop to receive shapes.
        shape_imgs: Rendered shape images to paste.
        shape_params: Shape metadata paired with each rendered shape.
        blur_radius: Blur radius used by the generation pipeline.

    Returns:
        Bounding boxes and updated background image.
    """
    shape_bboxes: List[Tuple[int, int, int, int, int]] = []

    for i, shape_param in enumerate(shape_params):

        x = shape_param[-2]
        y = shape_param[-1]
        shape_img = shape_imgs[i]
        x1, y1, x2, y2 = shape_img.getbbox()
        bg_at_shape = background.crop((x1 + x, y1 + y, x2 + x, y2 + y))
        bg_at_shape.paste(shape_img, (0, 0), shape_img)
        bg_at_shape = bg_at_shape.filter(ImageFilter.MedianFilter(3))
        background.paste(bg_at_shape, (x, y))

        im_w, im_h = background.size
        x /= im_w
        y /= im_h

        w = (x2 - x1) / im_w
        h = (y2 - y1) / im_h

        shape_bboxes.append((CLASSES.index(shape_param[0]), x, y, w, h))
        """
        shape_bboxes.append(
            (
                CLASSES.index(shape_param[2]),
                x + (0.1 * w),
                y + (0.1 * h),
                0.8 * w,
                0.8 * h,
            )
        )
        """
    return shape_bboxes, background.convert("RGB")


def get_backgrounds() -> List[pathlib.Path]:
    """Get a list of all the background images.

    Returns:
        Paths to all configured background images.
    """
    # Cover all the necessary extensions
    exts, filenames = ["png", "jpg", "jpeg"], []
    for backgrounds_folder in config.BACKGROUNDS_DIRS:
        for ext in exts:
            filenames.extend(list(backgrounds_folder.rglob(f"*.{ext}")))
            filenames.extend(list(backgrounds_folder.rglob(f"*.{ext.upper()}")))

    print(f"Found {len(filenames)} backgrounds.")

    return filenames


def get_base_shapes(shape):
    """Get base shape images for a shape type.

    Args:
        shape: Shape name to load.

    Returns:
        Loaded base shape images.
    """
    # For now just using the first one to prevent bad alpha placement
    return [
        Image.open(base_path)
        for base_path in (config.BASE_SHAPES_DIRS[0] / shape).glob("*.png")
    ]


def random_list(items, count):
    """Get a randomly sampled list.

    Args:
        items: Items to sample from.
        count: Number of values to return.

    Returns:
        Random selections from ``items``.
    """
    return [random.choice(items) for i in range(0, count)]


def create_shape(
    shape,
    base,
    alpha,
    font_file,
    size,
    angle,
    target_color,
    target_rgb,
    alpha_color,
    alpha_rgb,
    x,
    y,
) -> PIL.Image.Image:
    """Create one rendered synthetic target shape.

    Args:
        shape: Shape name.
        base: Base image for the shape.
        alpha: Alphanumeric character to draw.
        font_file: Font path for the alphanumeric.
        size: Target size in pixels.
        angle: Rotation angle.
        target_color: Named target color.
        target_rgb: RGB target color.
        alpha_color: Named alphanumeric color.
        alpha_rgb: RGB alphanumeric color.
        x: Target x coordinate.
        y: Target y coordinate.

    Returns:
        Rendered RGBA target image.
    """

    image = get_base(base, target_rgb, size)
    image = strip_image(image)

    image = add_alphanumeric(image, shape, alpha, alpha_rgb, font_file)

    w, h = image.size
    ratio = min(size / w, size / h)
    image = image.resize((int(w * ratio), int(h * ratio)), 1)

    image = rotate_shape(image, angle)
    image = strip_image(image)

    return image


def get_base(base, target_rgb, size):
    """Copy and recolor a base shape.

    Args:
        base: Source base shape image.
        target_rgb: RGB color to apply to the shape.
        size: Requested target size.

    Returns:
        Recolored base image.
    """
    return image_ops.recolor_nonwhite(base.copy().resize((256, 256), 1), target_rgb)


def strip_image(image: PIL.Image.Image) -> PIL.Image.Image:
    """Remove white and black edges.

    Args:
        image: Image to strip.

    Returns:
        Cropped transparent image.
    """
    return image_ops.transparent_white(image, threshold=247)

def add_alphanumeric(
    image: PIL.Image.Image,
    shape: str,
    alpha,
    alpha_rgb: Tuple[int, int, int],
    font_file,
) -> PIL.Image.Image:
    """Draw an alphanumeric character on a target shape.

    Args:
        image: Target shape image.
        shape: Shape name.
        alpha: Alphanumeric character.
        alpha_rgb: RGB text color.
        font_file: Font path.

    Returns:
        Image with alphanumeric text.
    """

    font_multiplier = random.uniform(*_ALPHA_INFO[shape].font_multiplier)

    # Set font size, select font style from fonts file, set font color
    font_size = int(round(font_multiplier * image.height))
    font = ImageFont.truetype(str(font_file), font_size)
    draw = ImageDraw.Draw(image)

    w, h = draw.textsize(alpha, font=font)

    x = (image.width - w) / 2
    y = (image.height - h) / 2

    dx, dy = _ALPHA_OFFSETS.get(shape, (0, 0))
    x += dx
    y += dy

    draw.text((x, y), alpha, alpha_rgb, font=font)

    return image


def rotate_shape(image, angle):
    """Rotate a target shape image.

    Args:
        image: Image to rotate.
        angle: Rotation angle in degrees.

    Returns:
        Rotated image.
    """
    return image.rotate(angle, expand=1)


def create_coco_metadata(data_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    """Create COCO metadata from generated per-image labels.

    Args:
        data_dir: Directory containing generated images and JSON labels.
        out_path: Output COCO metadata path.
    """
    images, annotations = [], []
    jsons = sorted(list(data_dir.glob("*.json")))
    for label_file in jsons:
        labels = _load_labels(label_file)
        images.append(_coco_image(label_file, labels))
        annotations.extend(_coco_annotations(labels, len(annotations)))
    out_path.write_text(
        json.dumps(
            {
                "categories": _coco_categories(),
                "images": images,
                "annotations": annotations,
            },
            indent=2,
        )
    )


def _load_labels(label_file: pathlib.Path) -> dict:
    return json.loads(label_file.read_text())


def _coco_categories() -> List[dict]:
    return [
        {"supercategory": "none", "name": name, "id": idx}
        for idx, name in enumerate(CLASSES)
    ]


def _coco_image(label_file: pathlib.Path, labels: dict) -> dict:
    return {
        "file_name": label_file.with_suffix(f"{config.IMAGE_EXT}").name,
        "width": config.DETECTOR_SIZE[0],
        "height": config.DETECTOR_SIZE[1],
        "id": labels["image_id"],
    }


def _coco_annotations(labels: dict, start_id: int) -> List[dict]:
    annotations = []
    for idx, label in enumerate(labels["bboxes"]):
        annotations.append(_coco_annotation(label, labels["image_id"], start_id + idx))
    return annotations


def _coco_annotation(label: dict, image_id: int, annotation_id: int) -> dict:
    x1, y1, w, h = _coco_box(label)
    return {
        "id": annotation_id,
        "bbox": [x1, y1, w, h],
        "category_id": label["class_id"],
        "iscrowd": 0,
        "area": w * h,
        "image_id": image_id,
        "segmentation": [[x1, y1, x1, y1 + h, x1 + w, y1 + h, x1 + w, y1]],
    }


def _coco_box(label: dict) -> Tuple[int, int, int, int]:
    return (
        int(label["x1"] * config.DETECTOR_SIZE[0]),
        int(label["y1"] * config.DETECTOR_SIZE[1]),
        int(label["w"] * config.DETECTOR_SIZE[0]),
        int(label["h"] * config.DETECTOR_SIZE[1]),
    )


if __name__ == "__main__":
    # Pull the assets if not present locally.
    asset_manager.pull_all()
    generate_all_images(
        config.DATA_DIR / "detector_train",
        config.DET_TRAIN_IMAGES,
        config.DET_TRAIN_OFFSET,
    )
    generate_all_images(
        config.DATA_DIR / "detector_val", config.DET_VAL_IMAGES, config.DET_VAL_OFFSET,
    )
