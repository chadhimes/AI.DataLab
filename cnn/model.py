import torch.nn as nn
import torch
from torchvision import models


def build_resnet18(pretrained: bool = True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model
