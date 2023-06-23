from torchvision import datasets
from torch.utils.data import DataLoader


def get_data_loader(train_transforms, test_transforms, batch_size=64):
    """Gets instance of train and test loader of CIFAR 10 Dataset"""
    
    train = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    test = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)

    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader