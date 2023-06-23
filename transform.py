import torch
from torchvision import transforms


def get_transforms():
    """Gets instance of train and test transforms"""

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, .4468), (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, .4468), (0.2470, 0.2435, 0.2616))
    ])

    return train_transform, test_transform