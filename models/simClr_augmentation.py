import torchvision.transforms as T

class SimCLRTransform:
    def __init__(self, size=32):
        self.t = T.Compose([
            T.ToTensor(),
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=max(3, int(0.1 * size)) if size >= 32 else 3),

            # If using ImageNet-pretrained ResNet, resize to 224 & normalize accordingly.
            # For CIFAR-sized ResNet, keep 32x32 and use CIFAR stats or none.
        ])

    def __call__(self, x):
        return self.t(x), self.t(x)