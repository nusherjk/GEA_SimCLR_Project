import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_cifar10(batch_size=64, flatten=True):
    """
    Loads the CIFAR-10 dataset with optional flattening (for MLPs).

    Args:
        batch_size (int): Number of samples per batch.
        flatten (bool): Whether to flatten the images (use True for MLPs).

    Returns:
        train_loader, test_loader: DataLoaders for training and testing.
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader