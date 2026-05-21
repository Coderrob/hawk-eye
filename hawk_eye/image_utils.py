"""Shared image tiling helpers."""

from typing import Iterable, Tuple


def _tile_step(image_length: int, tile_length: int, overlap: int) -> int:
    return image_length if image_length == tile_length else tile_length - overlap


def _tile_origin(origin: int, image_length: int, tile_length: int) -> int:
    return 0 if origin == 0 else min(origin, image_length - tile_length)


def _axis_origins(image_length: int, tile_length: int, overlap: int) -> Iterable[int]:
    step = _tile_step(image_length, tile_length, overlap)
    origins = range(0, image_length - overlap, step)
    return (_tile_origin(origin, image_length, tile_length) for origin in origins)


def tile_origins(
    width: int, height: int, tile_size: Tuple[int, int], overlap: int
) -> Iterable[Tuple[int, int]]:
    """Yield top-left tile origins that cover an image.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        tile_size: Tile width and height.
        overlap: Pixel overlap between adjacent tiles.

    Yields:
        ``(x, y)`` tile origins.
    """
    for x in _axis_origins(width, tile_size[0], overlap):
        for y in _axis_origins(height, tile_size[1], overlap):
            yield x, y
