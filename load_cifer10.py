import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T
def get_simclr_transform(size=32):
    return T.Compose([
        T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=max(3, int(0.1 * size)) if size >= 32 else 3),
        T.ToTensor(),
    ])
class SimCLRDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2, label


def load_cifar10(batch_size=32, flatten=False):
    """
    Loads the CIFAR-10 dataset with optional flattening (for MLPs).

    Args:
        batch_size (int): Number of samples per batch.
        flatten (bool): Whether to flatten the images (use True for MLPs).

    Returns:
        train_loader, test_loader: DataLoaders for training and testing.
    """
    transform_list = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # transform = transforms.Compose([
    #     transforms.Resize(224),  # ResNet expects 224x224
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)


    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
def load_cifar10_f_jacobian(batch_size=32, flatten=False):
    """
    Loads the CIFAR-10 dataset with optional flattening (for MLPs).

    Args:
        batch_size (int): Number of samples per batch.
        flatten (bool): Whether to flatten the images (use True for MLPs).

    Returns:
        train_loader, test_loader: DataLoaders for training and testing.
    """
    transform_list = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # transform = transforms.Compose([
    #     transforms.Resize(224),  # ResNet expects 224x224
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    transform_f_simClr = get_simclr_transform(size=32)
    cifar10_train  = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset = SimCLRDataset(cifar10_train, transform_f_simClr)
    # test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader