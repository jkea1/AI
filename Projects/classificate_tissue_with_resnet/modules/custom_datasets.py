import numpy as np
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
        img = self.images[i].astype("uint8")
        label = self.labels[i]

        if self.transform:
            img = Image.fromarray(img) # PyTorch의 transforms는 PIL.Image 또는 Tensor 타입의 입력을 기대한다.
            img = self.transform(img)

        return img, label