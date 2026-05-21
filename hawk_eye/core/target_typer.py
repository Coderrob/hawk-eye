""" """

import yaml

import torch

from hawk_eye.core import pull_assets
from third_party.efficientdet import efficientnet
from third_party.vovnet import vovnet


class TargetTyper(torch.nn.Module):
    def __init__(
        self,
        version: str = None,
        backbone: str = None,
        use_cuda: bool = False,
        half_precision: bool = False,
    ) -> None:
        super().__init__()
        self.use_cuda = use_cuda
        self.half_precision = half_precision
        self._validate_model_source(version, backbone)
        self.model = self._load_model(version, backbone)
        self._prepare_model()

    def _validate_model_source(self, version: str, backbone: str) -> None:
        if backbone is None and version is None:
            raise ValueError("Must supply either model version or backbone to load")

    def _load_model(self, version: str, backbone: str) -> torch.nn.Module:
        if version is None:
            return self._load_backbone(backbone)
        return self._load_versioned_model(version)

    def _load_versioned_model(self, version: str) -> torch.nn.Module:
        model_path = pull_assets.download_model(
            model_type="target_typer", version=version
        )
        config = yaml.safe_load((model_path.parent / "config.yaml").read_text())
        model = self._load_backbone(config.get("model", {}).get("backbone", None))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model

    def _prepare_model(self) -> None:
        self.model.eval()

        if self.use_cuda and self.half_precision:
            self.model.cuda()
            self.model.half()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.final_features(x)

    def _load_backbone(self, backbone: str) -> torch.nn.Module:
        """ Load the supplied backbone. """
        if "efficientnet" in backbone:
            model = efficientnet.EfficientNet(backbone=backbone, num_classes=1)
        elif "vovnet" in backbone:
            model = vovnet.VoVNet(backbone, num_classes=1)
        else:
            raise ValueError(f"Unsupported backbone {backbone}.")

        model.delete_classification_head()

        return model
