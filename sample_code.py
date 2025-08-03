import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
import copy
import numpy as np
from torch.utils.data import DataLoader, random_split

# ----------------------------- Load CIFAR-10 -----------------------------
def load_cifar10(batch_size=128, flatten=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1)) if flatten else transforms.Lambda(lambda x: x)
    ])

    trainval = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(trainval))
    val_size = len(trainval) - train_size
    train, val = random_split(trainval, [train_size, val_size])

    return DataLoader(train, batch_size=batch_size, shuffle=True), \
           DataLoader(val, batch_size=batch_size, shuffle=False), \
           DataLoader(test, batch_size=batch_size, shuffle=False)

# ------------------------ Dynamic MLP Model Builder ------------------------
class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=10, activation='relu'):
        super().__init__()
        activations = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
        self.activation = activations[activation]
        layers = []
        prev = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(copy.deepcopy(self.activation))
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------- Fitness Evaluation -------------------------
def evaluate_fitness(candidate, train_loader, val_loader, device='cpu'):
    input_size = 32 * 32 * 3
    hidden_layers, lr, act = candidate['layers'], candidate['lr'], candidate['act']
    model = DynamicMLP(input_size, hidden_layers, activation=act).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):  # Short training for fitness estimate
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            correct += (pred.argmax(1) == yb).sum().item()
            total += yb.size(0)

    return correct / total

# --------------------------- Evolution Logic ---------------------------
def initialize_population(size):
    population = []
    for _ in range(size):
        candidate = {
            'layers': random.choices([64, 128, 256], k=random.randint(1, 3)),
            'lr': random.choice([1e-3, 5e-4, 1e-4]),
            'act': random.choice(['relu', 'tanh'])
        }
        population.append(candidate)
    return population

def mutate(candidate):
    new = copy.deepcopy(candidate)
    if random.random() < 0.5:
        new['layers'] = random.choices([64, 128, 256], k=random.randint(1, 3))
    if random.random() < 0.5:
        new['lr'] = random.choice([1e-3, 5e-4, 1e-4])
    if random.random() < 0.5:
        new['act'] = random.choice(['relu', 'tanh'])
    return new

def crossover(parent1, parent2):
    child = {
        'layers': random.choice([parent1['layers'], parent2['layers']]),
        'lr': random.choice([parent1['lr'], parent2['lr']]),
        'act': random.choice([parent1['act'], parent2['act']])
    }
    return child

def guided_selection(population, fitnesses, top_k=5):
    sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: -pair[0])]
    return sorted_pop[:top_k]

# --------------------------- Main GEA Loop ---------------------------
def run_gea(generations=10, pop_size=10, device='cpu'):
    train_loader, val_loader, _ = load_cifar10()
    population = initialize_population(pop_size)

    for gen in range(generations):
        print(f"\nGeneration {gen}")
        fitnesses = [evaluate_fitness(c, train_loader, val_loader, device=device) for c in population]

        top_candidates = guided_selection(population, fitnesses)
        best_fit = max(fitnesses)
        print(f"Best fitness: {best_fit:.4f}")

        next_gen = top_candidates.copy()

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(top_candidates, 2)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)

        population = next_gen

    return top_candidates[0]  # Return the best architecture

# --------------------------- Run Script ---------------------------
if __name__ == "__main__":
    best_model_config = run_gea(generations=5, pop_size=8, device='cuda' if torch.cuda.is_available() else 'cpu')
    print("\nBest evolved model config:")
    print(best_model_config)
