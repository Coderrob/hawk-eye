""" A FPN similar to the one defined in Dectectron2:
https://github.com/facebookresearch/detectron2
"""

from typing import List
import collections

import torch


def depthwise(in_channels: int, out_channels: int) -> torch.nn.Module:
    """Build a depthwise separable convolution block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        Depthwise separable convolution block.
    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=in_channels,
        ),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
    )


def conv3x3(in_channels: int, out_channels: int) -> torch.nn.Module:
    """Build a standard 3x3 convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        Standard 3x3 convolution layer.
    """
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1, bias=True
    )


def _conv_factory(use_dw: bool):
    return depthwise if use_dw else conv3x3


class FPN(torch.nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_levels: int = 5,
        use_dw: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_in = len(in_channels)
        self.num_levels = num_levels

        conv = _conv_factory(use_dw)

        # Construct the lateral convolutions to adapt the incoming feature maps
        # to the same channel depth.
        self.lateral_convs = torch.nn.ModuleList([])
        for channels in reversed(in_channels):
            self.lateral_convs.append(
                torch.nn.Conv2d(channels, out_channels, kernel_size=1)
            )

        # Construct a convolution per level.
        self.convs = torch.nn.ModuleList([])
        for _ in range(self.num_levels):
            self.convs.append(conv(out_channels, out_channels))

    def __call__(
        self, feature_maps: collections.OrderedDict
    ) -> collections.OrderedDict:
        """Take the input feature maps and apply the necessary convolutions to build
        out the num_levels specified."""
        self._apply_lateral_levels(feature_maps)
        self._append_extra_levels(feature_maps)
        return feature_maps

    def _apply_lateral_levels(
        self, feature_maps: collections.OrderedDict
    ) -> None:
        for idx, level_idx in enumerate(reversed(feature_maps.keys())):
            feature_maps[level_idx] = self.lateral_convs[idx](feature_maps[level_idx])
            if level_idx < next(reversed(feature_maps.keys())):
                feature_maps[level_idx] += self._upsample_previous(
                    feature_maps, level_idx
                )
            feature_maps[level_idx] = self.convs[idx](feature_maps[level_idx])

    def _upsample_previous(
        self, feature_maps: collections.OrderedDict, level_idx: int
    ) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            feature_maps[level_idx + 1],
            feature_maps[level_idx].shape[2:],
            align_corners=True,
            mode="bilinear",
        )

    def _append_extra_levels(
        self, feature_maps: collections.OrderedDict
    ) -> None:
        for idx in range(self.num_levels - self.num_in):
            new_id = next(reversed(feature_maps.keys())) + 1
            feature_maps[new_id] = self._new_extra_level(feature_maps, new_id, idx)

    def _new_extra_level(
        self, feature_maps: collections.OrderedDict, new_id: int, idx: int
    ) -> torch.Tensor:
        new_level = torch.nn.functional._max_pool2d(
            feature_maps[new_id - 1], kernel_size=3, stride=2, padding=1
        )
        new_level = self.convs[self.num_in + idx](new_level)
        if idx != (self.num_levels - self.num_in - 1):
            return torch.nn.functional.relu(new_level, inplace=True)
        return new_level
