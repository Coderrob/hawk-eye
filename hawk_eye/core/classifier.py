"""A classifier model which wraps around a backbone. This setup allows for easy
interchangeability during experimentation and a reliable way to load saved models."""

import pathlib
from typing import Optional

import torch
import yaml

try:
    from hawk_eye.core import asset_manager
except ModuleNotFoundError:
    pass
from third_party.rexnet import rexnet
from third_party.vovnet import vovnet


class Classifier(torch.nn.Module):
    def __init__(
        self,
        num_classes: Optional[int] = 2,
        timestamp: Optional[str] = None,
        backbone: Optional[str] = None,
        half_precision: Optional[bool] = False,
    ) -> None:
        """
        Args:
            num_classes: The number of classes to predict.
            timestamp: The timestamp of the model to download from GCloud.
            backbone: A string designating which model to load.
            half_precision: Whether to use half precision. This should be False for
                training but True during inference.

        :raises ValueError: Error if neither a timestamp or backbone arg is supplied.

        Examples:
            >>> classifier = Classifier(2, backbone="vovnet-19")
            >>> with torch.no_grad():
            ...    predictions = classifier.classify(torch.randn(1, 3, 64, 64), True)
            >>> predictions.shape
            torch.Size([1, 2])
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_cuda = torch.cuda.is_available()
        self.half_precision = half_precision and self.use_cuda
        self._validate_model_source(timestamp, backbone)
        self.model = self._load_model(timestamp, backbone)
        self._prepare_model()

    def _validate_model_source(self, timestamp: str, backbone: str) -> None:
        if backbone is None and timestamp is None:
            raise ValueError("Must supply either model timestamp or backbone to load")

    def _load_model(self, timestamp: str, backbone: str) -> torch.nn.Module:
        if timestamp is None:
            return self.load_backbone(backbone)
        return self._load_versioned_model(timestamp)

    def _load_versioned_model(self, timestamp: str) -> torch.nn.Module:
        model_path = self._model_path(timestamp)
        config = yaml.safe_load((model_path / "config.yaml").read_text())["model"]
        model = self.load_backbone(config.get("backbone", None))
        self.load_state_dict(torch.load(model_path / "classifier.pt", map_location="cpu"))
        self.image_size = config["image_size"]
        return model

    def _model_path(self, timestamp: str) -> pathlib.Path:
        production_models = pathlib.Path(__file__).parent / "production_models"
        if production_models.is_dir():
            return production_models / timestamp
        return asset_manager.download_model("classifier", timestamp)

    def _prepare_model(self) -> None:
        self.model.eval()

        if self.use_cuda:
            self.model.cuda()

        if self.half_precision:
            self.model.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier.

        Args:
            x: input tensor.

        Returns:
            the output tensor.
        """
        # If half precision, assume inference not train.
        if self.half_precision:
            x = x.half()

        return self.model(x)

    def load_backbone(self, backbone: str) -> torch.nn.Module:
        """Load the supplied backbone. See this function for the list of potential
        backbones that can be loaded.

        Args:
            backbone: The backbone type to load.

        Returns:
            The loaded model.

        :raises ValueError: If improper backbone is supplied.
        """
        if "rexnet" in backbone:
            model = rexnet.ReXNet(num_classes=self.num_classes, model_type=backbone)
        elif "vovnet" in backbone:
            model = vovnet.VoVNet(model_name=backbone, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported backbone {backbone}.")

        return model

    def classify(self, x: torch.Tensor, probability: bool = False) -> torch.Tensor:
        """Take in an image batch and return the class for each image. If specified,
        softmax will be applied to the predictions.

        Args:
            x: Input tensor of size (batch, height, width, channels).
            probability: Whether or not to apply softmax.

        Returns:
            The output tensor.
        """

        if self._should_half_inputs():
            x = x.half()
        if probability:
            return torch.nn.functional.softmax(self.model(x), dim=1)
        _, predicted = torch.max(self.model(x).data, 1)
        return predicted

    def _should_half_inputs(self) -> bool:
        return self.use_cuda and self.half_precision
