from load_cifer10 import load_cifar10
from models.simClr_augmentation import SimCLRTransform
# Load CIFAR-10 and make two augmented views per image
x, y = load_cifar10()
x1 , x2 = SimCLRTransform(x)





# Encode each view with the backbone (ResNet)


# Project features with the projection head (MLP)



# Compute the contrastive (NT-Xent) loss



# Backprop + update



