import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PanNukeDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i] # 이미 float32, (3, 224, 224)
        label = self.labels[i]

        img = torch.from_numpy(img) # numpy → tensor

        if self.transform:
            img = self.transform(img) # 예: Normalize

        return img, label