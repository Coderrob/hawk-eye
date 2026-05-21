""" Datasets for loading data for our various training regimes. """

from typing import List, Tuple
import pathlib
import json
import cv2
import torch

from hawk_eye.train import augmentations as augs


class DetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        metadata_path: pathlib.Path,
        img_ext: str = ".png",
        img_width: int = 512,
        img_height: int = 512,
        validation: bool = False,
    ) -> None:
        super().__init__()
        self.meta_data = json.loads(metadata_path.read_text())
        self.images = list(data_dir.glob(f"*{img_ext}"))
        assert self.images, f"No images found in {data_dir}."

        self.img_height = img_height
        self.img_width = img_width
        self.len = len(self.images)
        self.transform = (
            augs.det_val_augs(img_height, img_width)
            if validation
            else augs.det_train_augs(img_height, img_width)
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.images[idx]))
        assert image is not None, f"Trouble reading {self.images[idx]}."
        labels = json.loads(self.images[idx].with_suffix(".json").read_text())
        boxes = self._boxes(labels)
        category_ids = [label["class_id"] for label in labels["bboxes"]]

        return self.transform(
            image=image,
            bboxes=boxes,
            category_ids=category_ids,
            image_ids=labels["image_id"],
        )

    def _boxes(self, labels: dict) -> List[torch.Tensor]:
        boxes = [
            torch.Tensor(
                [item["x1"], item["y1"], item["x1"] + item["w"], item["y1"] + item["h"]]
            )
            for item in labels["bboxes"]
        ]

        if boxes:
            boxes = torch.stack(boxes).clamp(0.0, 1.0)
        return boxes

    def __len__(self) -> int:
        return self.len
