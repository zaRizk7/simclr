from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import DatasetFolder

__all__ = ['SimCLRDataset']

SUPPORTED_EXTENSION = ["png", "jpg", "jpeg"]

DEFAULT_TRANSFORM = T.Compose(
    [T.RandAugment(), T.RandomResizedCrop((224, 224)), T.ToTensor()]
)


class SimCLRDataset:
    def __init__(
        self, root, transform_1=DEFAULT_TRANSFORM, transform_2=DEFAULT_TRANSFORM
    ):
        self.data = self._build(root)
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def _build(self, root):
        return [
            str(f)
            for ext in SUPPORTED_EXTENSION
            for f in Path(root).glob(f"**/*.{ext}")
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = Image.open(sample)

        image_1 = self.transform_1(image)
        image_2 = self.transform_2(image)

        return image_1, image_2
