#!/usr/bin/env python3
""" This script generates training data for the object
detector model. The output will be directories of images
plus COCO metadata jsons. """

from typing import List, Tuple
import multiprocessing
import random
import itertools

from tqdm import tqdm
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from hawk_eye.data_generation import generate_config as config
from hawk_eye.data_generation import image_ops
from hawk_eye.core import pull_assets

# Get constants from config
NUM_GEN = int(config.NUM_IMAGES)
MAX_SHAPES = int(config.MAX_PER_SHAPE)
FULL_SIZE = config.FULL_SIZE
TARGET_COLORS = config.TARGET_COLORS
ALPHA_COLORS = config.ALPHA_COLORS
COLORS = config.COLORS
CLASSES = config.OD_CLASSES
ALPHAS = config.ALPHAS

_NUM_COMBINATIONS = 1000
_FONT_MULTIPLIERS = {
    "star": 0.14,
    "triangle": 0.5,
    "rectangle": 0.72,
    "quarter-circle": 0.60,
    "semicircle": 0.55,
    "circle": 0.55,
    "square": 0.60,
    "trapezoid": 0.60,
}
_ALPHA_OFFSETS = {
    "trapezoid": (0, -20),
    "triangle": (-24, 12),
    "quarter-circle": (14, -40),
    "cross": (0, -25),
    "square": (0, -10),
}


def generate_all_images(gen_type: str, num_gen: int, offset: int = 0) -> None:
    """Generate all combinations of shape, shape color, alpha, and alpha color.

    Args:
        gen_type: Dataset split or folder name to generate.
        num_gen: Requested number of images.
        offset: Starting image index offset.
    """
    images_dir = config.DATA_DIR / gen_type
    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
    images_dir.mkdir(exist_ok=True, parents=True)

    r_state = random.getstate()
    random.seed(f"{gen_type}{offset}")

    base_shapes = {shape: get_base_shapes(shape) for shape in config.SHAPE_TYPES}

    # There are multiple paramters this data must map.
    # Shape base, shape color, alpha-numeric, and alpha-numeric color
    a = [
        config.SHAPE_TYPES,
        TARGET_COLORS,
        config.ALPHAS,
        ALPHA_COLORS,
        [angle for angle in range(0, 360, 45)],
    ]
    combinations = list(itertools.product(*a))
    random.shuffle(combinations)

    # Assume that 0 combinations means all
    _NUM_COMBINATIONS = 3000
    if _NUM_COMBINATIONS == 0:
        _NUM_COMBINATIONS = len(combinations)

    combinations = combinations[:_NUM_COMBINATIONS]
    num_gen = len(combinations)

    numbers = list(range(offset, num_gen + offset))
    crop_xs = random_list(range(0, config.FULL_SIZE[0] - config.CROP_SIZE[0]), num_gen)
    crop_ys = random_list(range(0, config.FULL_SIZE[1] - config.CROP_SIZE[1]), num_gen)

    backgrounds = random_list(get_backgrounds(), num_gen)
    num_targets = 1
    shape_params = []

    for combination in combinations[:_NUM_COMBINATIONS]:
        combination = list(combination)
        font_files = random_list(config.ALPHA_FONTS, num_targets)

        # Make sure shape and alpha are different colors
        if combination[1] == combination[3]:
            while combination[3] == combination[1]:
                combination[3] = random.choice(ALPHA_COLORS)

        target_rgbs = [random.choice(COLORS[color]) for color in [combination[1]]]
        alpha_rgbs = [random.choice(COLORS[color]) for color in [combination[3]]]

        sizes = random_list(range(30, 65), num_targets)
        xs = random_list(range(65, config.CROP_SIZE[0] - 65, 20), num_targets)
        ys = random_list(range(65, config.CROP_SIZE[1] - 65, 20), num_targets)

        shape_params.append(
            list(
                zip(
                    [combination[0]],
                    base_shapes[combination[0]],
                    [combination[2]],
                    font_files,
                    sizes,
                    [combination[4]],
                    [combination[1]],
                    target_rgbs,
                    [combination[3]],
                    alpha_rgbs,
                    xs,
                    ys,
                )
            )
        )
    # Put everything into one large iterable so that we can split up
    # data across thread pools.
    data = zip(
        numbers, backgrounds, crop_xs, crop_ys, shape_params, [gen_type] * num_gen,
    )

    random.setstate(r_state)

    # Generate in a pool. If specificed, use a given number of threads.
    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(generate_single_example, data)
        for _ in tqdm(processes, total=num_gen):
            pass


def generate_single_example(data: zip) -> None:
    """Create one full image for the shape-combination dataset.

    Args:
        data: Packed generation parameters for one image.
    """
    (number, background, crop_x, crop_y, shape_params, gen_type) = data
    data_path = config.DATA_DIR / gen_type

    background = background.copy()
    background = background.crop(
        (crop_x, crop_y, crop_x + config.CROP_SIZE[0], crop_y + config.CROP_SIZE[1])
    )
    shape_imgs, img_name = create_shape(*shape_params[0])
    full_img = add_shapes(background, shape_imgs, shape_params)
    full_img = full_img.resize(config.DETECTOR_SIZE)

    img_fn = data_path / f"ex{number}_{img_name}{config.IMAGE_EXT}"
    full_img.save(img_fn)


def add_shapes(
    background: PIL.Image.Image, shape_img: PIL.Image.Image, shape_params,
) -> Tuple[List[Tuple[int, int, int, int, int]], PIL.Image.Image]:
    """Paste shapes onto a background image.

    Args:
        background: Background image to update.
        shape_img: Shape image to paste.
        shape_params: Metadata describing shape placement.

    Returns:
        Updated background image.
    """

    for i, shape_param in enumerate(shape_params):

        x = shape_param[-2]
        y = shape_param[-1]
        x1, y1, x2, y2 = shape_img.getbbox()
        bg_at_shape = background.crop((x1 + x, y1 + y, x2 + x, y2 + y))
        bg_at_shape.paste(shape_img, (0, 0), shape_img)
        background.paste(bg_at_shape, (x, y))
        # Slightly expand the bounding box in order to simulate variability with
        # the detection boxes. Always make the crop larger than needed because training
        # augmentations will only be able to crop down.
        dx = random.randint(0, int(0.1 * (x2 - x1)))
        dy = random.randint(0, int(0.1 * (y2 - y1)))
        x1 -= dx
        x2 += dx
        y1 -= dy
        y2 += dy

        background = background.crop((x1 + x, y1 + y, x2 + x, y2 + y))
        background = background.filter(ImageFilter.SMOOTH_MORE)
    return background.convert("RGB")


def get_backgrounds() -> List[Image.Image]:
    """Get the background assets.

    Returns:
        Loaded and resized background images.
    """
    # Can be a mix of .png and .jpg
    for backgrounds_folder in config.BACKGROUNDS_DIRS:
        filenames = list(backgrounds_folder.rglob("*.png"))
        filenames += list(backgrounds_folder.rglob("*.jpg"))

    return [Image.open(img).resize(config.FULL_SIZE) for img in filenames]


def get_base_shapes(shape) -> List[Image.Image]:
    """Get base shape images for a shape type.

    Args:
        shape: Shape name to load.

    Returns:
        Loaded base shape images.
    """
    # For now just using the first one to prevent bad alpha placement
    base_path = config.BASE_SHAPES_DIR / shape / f"{shape}-01.png"
    return [Image.open(base_path)]


def random_list(items, count) -> list:
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
    """Create one rendered shape-combination image.

    Args:
        shape: Shape name.
        base: Base image for the shape.
        alpha: Alphanumeric character.
        font_file: Font path.
        size: Requested target size.
        angle: Rotation angle.
        target_color: Named target color.
        target_rgb: RGB target color.
        alpha_color: Named alphanumeric color.
        alpha_rgb: RGB alphanumeric color.
        x: Target x coordinate.
        y: Target y coordinate.

    Returns:
        Rendered image and filename stem.
    """

    image = get_base(base, target_rgb, size)
    image = strip_image(image)
    image = add_alphanumeric(image, shape, alpha, alpha_rgb, font_file)

    w, h = image.size
    ratio = min(size / w, size / h)
    image = image.resize((int(w * ratio), int(h * ratio)), 1)

    image = rotate_shape(image, shape, angle)
    image = strip_image(image)
    img_name = f"{shape}_{target_color}_{alpha}_{alpha_color}_{angle}"
    return image, img_name


def get_base(base, target_rgb, size) -> Image.Image:
    """Copy and recolor the base shape.

    Args:
        base: Source base image.
        target_rgb: RGB color to apply.
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
    return image_ops.transparent_white(image, threshold=255)


def add_alphanumeric(
    image: PIL.Image.Image,
    shape: str,
    alpha,
    alpha_rgb: Tuple[int, int, int],
    font_file,
) -> PIL.Image.Image:
    """Draw an alphanumeric character on a shape.

    Args:
        image: Target shape image.
        shape: Shape name.
        alpha: Alphanumeric character.
        alpha_rgb: RGB text color.
        font_file: Font path.

    Returns:
        Image with alphanumeric text.
    """
    font_multiplier = _FONT_MULTIPLIERS.get(shape, 0.55)

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
    if shape == "circle":
        x -= random.randint(-15, 15)
        y -= random.randint(-15, 15)

    draw.text((x, y), alpha, alpha_rgb, font=font)

    return image


def rotate_shape(image, shape, angle) -> Image.Image:
    """Rotate a shape image.

    Args:
        image: Image to rotate.
        shape: Shape name.
        angle: Rotation angle in degrees.

    Returns:
        Rotated image.
    """
    return image.rotate(angle, expand=1)


if __name__ == "__main__":
    # Pull the assets if not present locally.
    pull_assets.pull_all()
    generate_all_images("combinations_train", config.NUM_IMAGES, config.NUM_OFFSET)
    generate_all_images(
        "combinations_val", config.NUM_VAL_IMAGES, config.NUM_VAL_OFFSET
    )
