import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import  datasets, models, transforms as T
from torch.utils.data import DataLoader, Dataset
from models.encoder_simCLR import Encoder
from models.mlp_projector import MLPProjector
from models.simClr_augmentation import SimCLRTransform
from torch.utils.tensorboard import SummaryWriter
import random
import math
import time




# -----------------------
# Dataset wrapper that yields two views per image
# -----------------------
class SimCLRDataset(Dataset):
    def __init__(self, root, train=True,  download=True):
        self.ds = datasets.CIFAR10(root=root, train=train, download=download)
        # self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        (x_i, x_j), y = self.ds[idx]
        # x1, x2 = self.transform(img)
        return x_i, x_j


# -----------------------
# NT-Xent loss (infoNCE)
# -----------------------
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cpu'):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        """
        z_i, z_j: projections of two views (B x D)
        returns scalar loss
        """
        B = z_i.shape[0]
        assert B == self.batch_size
        z = torch.cat([z_i, z_j], dim=0)  # 2B x D
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / self.temperature  # 2B x 2B

        # mask out self-similarity
        mask = (~torch.eye(2*B, 2*B, dtype=torch.bool)).to(self.device)

        sim_masked = sim.masked_select(mask).view(2*B, -1)  # 2B x (2B-1)

        positives = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0).unsqueeze(1)  # 2B x 1

        logits = torch.cat([positives, sim_masked], dim=1)  # 2B x (1 + 2B -1)
        labels = torch.zeros(2*B, dtype=torch.long).to(self.device)  # positives are first

        loss = self.criterion(logits, labels)
        loss = loss / (2 * B)
        return loss

def get_loader(batch_size=128):
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=SimCLRTransform(size=32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2  # ✅ works on Linux/macOS, may need 0 on Windows
    )
    return train_loader


def train_simclr(
    root="./data",
    epochs=100,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-6,
    temperature=0.5,
    projector_config=None,
    device=None,
    num_workers=4
):
    # writer = SummaryWriter(log_dir="runs/train_log{}")
    """
    Trains SimCLR on CIFAR-10 using your SimCLR(nn.Module).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data

    # train_ds = SimCLRDataset(root=root, train=True, download=True)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
    #                           num_workers=num_workers, drop_last=True)
    train_loader = get_loader(batch_size=batch_size)

    for batch in train_loader:
        print(type(batch[0]), type(batch[0][0]))
        break

    # Model
    if projector_config is None:
        projector_config = [
            {'in': 512, 'out': 2048, 'activation': 'relu', 'batchnorm': True},
            {'in': 2048, 'out': 128}  # final layer, projection
        ]
    # update
    encoder = Encoder().to(device)
    # feat_dim = encoder.out_dim
    projection_head = MLPProjector(projector_config).to(device)
    params = list(encoder.parameters()) + list(projection_head.parameters())
    # Optimizer
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Loss
    ntxent = NTXentLoss(batch_size=batch_size, temperature=temperature, device=device)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    for epoch in range(1, epochs + 1):
        encoder.train()
        projection_head.train()
        epoch_loss = 0.0
        for (x1, x2), y in train_loader:
            x1, x2 = x1.to(device), x2.to(device)

            optimizer.zero_grad()

            h1 = encoder(x1)
            h2 = encoder(x2)
            z1 = projection_head(h1)
            z2 = projection_head(h2)

            loss = ntxent(z1, z2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # writer.add_scalar("Loss/train", loss.item(), epoch)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
        # writer.close()


    return encoder, projection_head



# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # small run for verification
    model = train_simclr(
        root="./data",
        epochs=1,
        batch_size=128,
        lr=3e-4,
        weight_decay=1e-6,
        temperature=0.5,
        projector_config=None,  # use default
        device=None,
        num_workers=2,

    )
    # After training you can save models
    # torch.save(model.state_dict(), "simclr_model.pth")
    # model  = torch.load("simclr_model.pth")

    # evaluate(model, "cpu")
    # torch.save(projector.state_dict(), "simclr_projector.pth")
