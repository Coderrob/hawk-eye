"""Image operations shared by synthetic data generation scripts."""

from typing import Tuple

from PIL import Image


def recolor_nonwhite(image: Image.Image, rgb: Tuple[int, int, int]) -> Image.Image:
    """Recolor non-white pixels while preserving transparent background.

    Args:
        image: Source shape image.
        rgb: RGB color to apply to non-white pixels.

    Returns:
        Recolored RGBA image.
    """
    image = image.convert("RGBA")
    transparent = Image.new("RGBA", image.size, (0, 0, 0, 0))
    colored = Image.new("RGBA", image.size, (*rgb, 255))
    white_mask = image.convert("L").point(lambda value: 255 if value == 255 else 0)
    return Image.composite(transparent, colored, white_mask)


def transparent_white(image: Image.Image, threshold: int = 247) -> Image.Image:
    """Make near-white pixels transparent and crop to content.

    Args:
        image: Source image.
        threshold: Minimum channel value considered white.

    Returns:
        Cropped RGBA image with white pixels transparent.
    """
    image = image.convert("RGBA")
    alpha = image.convert("L").point(lambda value: 0 if value >= threshold else 255)
    image.putalpha(alpha)
    return image.crop(image.getbbox())
