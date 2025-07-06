import torch.nn as nn
from torchvision import models

def get_resnet(num_classes):
    # 미리 학습된 ResNet-18을 불러오고,
    # 마지막 출력층을 num_classes에 맞게 바꾼 후 반환.
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model