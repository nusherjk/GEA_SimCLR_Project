import pprint
import random

from load_cifer10 import load_cifar10_f_jacobian
from test_jacobianscore import load_cifar10_batch
import torch

from load_cifer10 import load_cifar10
from models.simclr import SimCLR
from torch import nn
import numpy as np
from test_jacobianscore import get_batch_jacobian, eval_score_perclass
from Evo_loop import  mutate_config, update_population
import torchvision
import torchvision.transforms as transforms
import json
device = 'cpu'
if  __name__ == '__main__':
    # Initialize 10 MLP Heads randomly from MlP Bench
    train_loader = load_cifar10_f_jacobian(64)
    with open('utils/mlp_bench.json') as f:
        mlp_bench = json.load(f)


    random_configurations = list()
    lower_range = 0
    upper_range = len(mlp_bench) -1

    positions = [np.random.randint(lower_range, upper_range+1) for _ in range(10)]
    for position in positions:

        arch = mlp_bench[position]

        model = SimCLR(arch)

        x , y = load_cifar10_batch(64)

        y = y.cpu().tolist()
        jacobs_batch = get_batch_jacobian(model, x)
        jacobs_batch = jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().tolist()
        try:
            score = eval_score_perclass(jacobs_batch, y)
        except Exception as e:
            print(e)
            score = np.nan



        random_configurations.append({
            "position":position,
            "mlp_conf": arch,
            "model": model,
            "score": score
        })


    # pprint.pprint(random_configurations)
    # Rank by highest score.
    sorted_random_configurations = sorted(random_configurations, key=lambda x: x["score"], reverse=True)
    # Select top 30%
    selected_population = sorted_random_configurations[:3]


    # pprint.pprint(selected_population)

    # evolve population for 2 generations
    for gen in range(2):

        print(f"\n=== Generation {gen} ===")
        new_gens = list()
        for population in selected_population:
            new_gens += update_population(population['mlp_conf'], num_children=3)  # add 3 new configs for each population

        for i, cfg in enumerate(new_gens):
            print(f"Config {i}:")
            for layer in cfg:
                print("   ", layer)












