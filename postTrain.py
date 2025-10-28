import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import  datasets, transforms as T
from torch.utils.data import DataLoader, Dataset

EPOCH = 100
def post_train_linearClassifier(model, device):

    transform = T.Compose([
        T.ToTensor()
    ])
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Extract features
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            h = model(x)
            features.append(h.cpu())
            labels.append(y)
    features = torch.cat(features, dim=0).to(device)
    labels = torch.cat(labels, dim=0).to(device)

    # Train linear classifier
    clf = nn.Linear(features.size(1), 10).to(device)
    optimizer = optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):  # small linear probe
        for i in range(0, len(features), 128):
            xb = features[i:i+128]
            yb = labels[i:i+128]
            preds = clf(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Compute accuracy
    with torch.no_grad():
        preds = clf(features).argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    print(f"Linear evaluation accuracy: {acc*100:.2f}%")
    return acc*100

