"""Submission ResNet-34 untuk Studio 2."""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5

def get_model():
    # Menggunakan ResNet-34 agar sesuai dengan bobot terbaik model_best.pth
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def get_transform():
    # Transformasi standar untuk validasi dan pengujian
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
