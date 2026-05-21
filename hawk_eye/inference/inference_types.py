"""Contains basic storage types passed around in the library."""

import enum
from typing import Optional

from PIL import Image


def _ranges_overlap(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> bool:
    return max(first_start, second_start) <= min(first_end, second_end)


@enum.unique
class Color(enum.Enum):
    """Contains colors for the AUVSI SUAS Interop Server.

    These colors can be used for both the background color and the
    alphanumeric color.

    .. note::
        :attr:`NONE` can be used if a color cannot be identified.
    """

    NONE = 0
    WHITE = 1
    BLACK = 2
    GRAY = 3
    RED = 4
    BLUE = 5
    GREEN = 6
    YELLOW = 7
    PURPLE = 8
    BROWN = 9
    ORANGE = 10


@enum.unique
class Shape(enum.Enum):
    """Contains target shapes for the AUVSI SUAS Interop Server.

    .. note::
        :attr:`NAS` (not-a-shape) can be used if a shape cannot be identified.
    """

    NAS = 0
    CIRCLE = 1
    SEMICIRCLE = 2
    QUARTER_CIRCLE = 3
    TRIANGLE = 4
    SQUARE = 5
    RECTANGLE = 6
    TRAPEZOID = 7
    PENTAGON = 8
    HEXAGON = 9
    HEPTAGON = 10
    OCTAGON = 11
    STAR = 12
    CROSS = 13


class Target:
    """Represents a target found on an image.

    This is intended to be built upon as the the target is being
    classified. Note that a target must have at least an x-position, y-position,
    width, and height to be created. A target should only be returned back by the
    library if at also contains a background color as well."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        shape: Optional[Shape] = Shape.NAS,
        orientation: Optional[float] = 0.0,
        background_color: Optional[Color] = Color.NONE,
        alphanumeric: Optional[str] = "",
        alphanumeric_color: Optional[Color] = Color.NONE,
        image: Optional[Image.Image] = None,
        confidence: Optional[float] = 0.0,
    ) -> None:
        """

        Args:
            x: The x position of the top-left corner in pixels.
            y: The y position of the top-left corner in pixels.
            width: The width of the blob in pixels.
            height: The height of the blob in pixels.
            orientation: The orientation of the target. An
                orientation of 0 means the target is not rotated, an
                orientation of 90 means it the top of the target points
                to the right of the image (0 <= orientation < 360).
            shape: The target :class:`Shape`.
            background_color: The target background :class:`Color`.
            alphanumeric: The letter(s) and/or number(s) on the
                target. May consist of one or more of the characters 0-9,
                A-Z, a-z. Typically, this will only be one capital
                letter.
            alphanumeric_color: The target alphanumeric :class:`Color`.
            image: Image showing the target.
            confidence: The confidence that the target exists
                (0 <= confidence <= 1).
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.shape = shape
        self.orientation = orientation
        self.background_color = background_color
        self.alphanumeric = alphanumeric
        self.alphanumeric_color = alphanumeric_color
        self.image = image
        self.confidence = confidence

    def overlaps(self, other_target: Shape) -> bool:
        """Check if another :class:`Target` has overlap.

        Args:
            other_target: The other :class:`Target` to compare with.

        Returns:
            ``True`` if there is overlap, ``False`` if no overlap.

        Examples::

            >>> target1 = Target(x=0, y=0, width=10, height=10)
            >>> target2 = Target(x=7, y=5, width=11, height=12)
            >>> target1.overlaps(target2)
            True
            >>> target1 = Target(x=0, y=0, width=10, height=10)
            >>> target2 = Target(x=20, y=22, width=30, height=32)
            >>> target1.overlaps(target2)
            False
        """
        x_overlap = _ranges_overlap(
            self.x,
            self.x + self.width,
            other_target.x,
            other_target.x + other_target.width,
        )
        y_overlap = _ranges_overlap(
            self.y,
            self.y + self.height,
            other_target.y,
            other_target.y + other_target.height,
        )
        return x_overlap and y_overlap

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (
            f"Target(x={self.x}, y={self.y}, width={self.width}, "
            f"height={self.height}, orientation={self.orientation}, "
            f"confidence={round(self.confidence, 2)}, shape={self.shape}, "
            f"background_color={self.background_color}, "
            f"alphanumeric={repr(self.alphanumeric)}, "
            f"alphanumeric_color={self.alphanumeric_color})"
        )
