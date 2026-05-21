"""A detector model which wraps around a feature extraction backbone, fpn, and RetinaNet
head.This allows for easy interchangeability during experimentation and a reliable way to
load saved models."""

import collections
import pathlib
from typing import List

import torch
import yaml

try:
    from hawk_eye.core import asset_manager
except ModuleNotFoundError:
    pass
from hawk_eye.core import fpn
from third_party.vovnet import vovnet
from third_party.detectron2 import postprocess
from third_party.detectron2 import regression
from third_party.detectron2 import anchors
from third_party.detectron2 import retinanet_head


class Detector(torch.nn.Module):
    def __init__(
        self,
        model_params: dict = None,
        timestamp: str = None,
        confidence: float = 0.05,
        num_detections_per_image: int = 100,
        half_precision: bool = False,
    ) -> None:
        super().__init__()
        self.half_precision = half_precision
        self.num_detections_per_image = num_detections_per_image
        self.confidence = confidence
        model_params, model_path = self._resolve_model_params(model_params, timestamp)

        self._build_model_components()
        self._prepare_cuda()
        self.image_size = model_params.get("image_size", 512)
        self.postprocess = self._create_postprocessor()
        self._load_weights(model_path)
        self.eval()

    def _resolve_model_params(
        self, model_params: dict, timestamp: str
    ) -> tuple:
        self._validate_model_source(model_params, timestamp)
        if timestamp is None:
            self._load_params(model_params)
            return model_params, None
        return self._load_timestamped_params(timestamp)

    def _validate_model_source(self, model_params: dict, timestamp: str) -> None:
        if model_params is None and timestamp is None:
            raise ValueError("Must supply either model timestamp or backbone to load")

    def _load_timestamped_params(self, timestamp: str) -> tuple:
        model_path = self._model_path(timestamp)
        config = yaml.safe_load((model_path / "config.yaml").read_text())
        self._load_params(config["model"])
        return config["model"], model_path

    def _model_path(self, timestamp: str) -> pathlib.Path:
        production_models = pathlib.Path(__file__).parent / "production_models"
        if production_models.is_dir():
            return production_models / timestamp
        return asset_manager.download_model("detector", timestamp)

    def _build_model_components(self) -> None:
        self.backbone = self._load_backbone(self.backbone)
        self.backbone.delete_classification_head()
        self.fpn = self._load_fpn(self.fpn_type, self.backbone.get_pyramid_channels())
        self._create_anchors()
        self._create_retinanet_head()

    def _create_anchors(self) -> None:
        assert len(self.anchor_sizes) == len(self.fpn_levels)
        self.anchors = anchors.AnchorGenerator(
            img_height=self.img_height,
            img_width=self.img_width,
            pyramid_levels=self.fpn_levels,
            aspect_ratios=self.aspect_ratios,
            sizes=self.anchor_sizes,
            anchor_scales=self.anchor_scales,
        )

    def _create_retinanet_head(self) -> None:
        self.retinanet_head = retinanet_head.RetinaNetHead(
            self.num_classes,
            in_channels=self.fpn_channels,
            anchors_per_cell=self.anchors.num_anchors_per_cell,
            num_convolutions=self.num_head_convs,
            use_dw=self.retinanet_head_dw,
        )

    def _prepare_cuda(self) -> None:
        if torch.cuda.is_available():
            self.anchors.all_anchors = self.anchors.all_anchors.cuda()
            self.anchors.anchors_over_all_feature_maps = [
                anchors.cuda() for anchors in self.anchors.anchors_over_all_feature_maps
            ]
            self.cuda()

    def _create_postprocessor(self) -> postprocess.PostProcessor:
        return postprocess.PostProcessor(
            num_classes=self.num_classes,
            image_size=self.image_size,
            all_anchors=self.anchors.all_anchors,
            regressor=regression.Regressor(),
            max_detections_per_image=self.num_detections_per_image,
            score_threshold=self.confidence,
            nms_threshold=0.2,
        )

    def _load_weights(self, model_path: pathlib.Path) -> None:
        if model_path is not None:
            self.load_state_dict(
                torch.load(model_path / "min-loss.pt", map_location="cpu")
            )

    def _load_params(self, config: dict) -> None:
        """Function to parse the model definition params for later building."""
        self.backbone = config.get("backbone")
        assert self.backbone is not None, "Please supply a backbone!"

        fpn_params = config.get("fpn")
        head_params = config.get("retinanet_head")
        assert fpn_params is not None, "Must supply a fpn section in the config."

        self.fpn_type = fpn_params.get("type")
        assert self.fpn_type is not None, "Must supply a fpn type."

        self.fpn_channels = fpn_params.get("num_channels", 128)
        self.fpn_use_dw = fpn_params.get("use_dw", False)
        self.num_head_convs = head_params.get("num_levels", 3)

        self.fpn_levels = fpn_params.get("levels", [3, 4, 5, 6, 7])
        self.retinanet_head_dw = head_params.get("use_dw")

        anchor_params = config.get("anchors")
        assert anchor_params is not None, "Please add an anchor section."
        self.aspect_ratios = anchor_params.get("aspect_ratios", [0.5, 1, 2])
        self.anchor_sizes = anchor_params.get("sizes", [16, 32, 64, 128, 256])
        self.anchor_scales = anchor_params.get("scales", [0.75, 1.0, 1.25])

        self.img_height, self.img_width = config.get("img_size", [512, 512])
        self.num_classes = config.get("num_classes", 10)

    def _load_backbone(self, backbone: str) -> torch.nn.Module:
        """Load the supplied backbone."""
        if "vovnet" in backbone:
            model = vovnet.VoVNet(backbone)
        else:
            raise ValueError(f"Unsupported backbone {backbone}.")

        return model

    def _load_fpn(self, fpn_name: str, features: List[int]) -> torch.nn.Module:
        if "retinanet" in fpn_name:
            fpn_ = fpn.FPN(
                in_channels=features[-3:],
                out_channels=self.fpn_channels,
                num_levels=len(self.fpn_levels),
                use_dw=self.fpn_use_dw,
            )

        return fpn_

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        levels = self.backbone.forward_pyramids(x)
        # Only keep the levels specified during construction.
        levels = collections.OrderedDict(
            [item for item in levels.items() if item[0] in self.fpn_levels]
        )
        levels = self.fpn(levels)
        classifications, regressions = self.retinanet_head(levels)

        if self.training:
            return classifications, regressions
        else:
            return self.postprocess(classifications, regressions)
