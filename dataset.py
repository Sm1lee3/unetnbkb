import os
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torch

class OxfordPetsDataset(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Завантаження датасету з сегментаційними масками
        self.dataset = OxfordIIITPet(
            root=self.root,
            target_types="segmentation",
            download=True
        )

        # Поділ на train / val (80/20)
        split_idx = int(0.8 * len(self.dataset))
        if split == "train":
            self.dataset = Subset(self.dataset, range(split_idx))
        elif split == "val":
            self.dataset = Subset(self.dataset, range(split_idx, len(self.dataset)))
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
            # НЕ дублюйте бінаризацію тут!

        return image, mask
