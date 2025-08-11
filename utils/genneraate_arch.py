import itertools
import json
import uuid

from MLP_bench import encode_arch
import torch
import numpy as np

# Generaate Arch generates the MLP benchmark configurations initially needed for creating random population




HIDDEN_SIZES = [64, 128, 256, 512]
ACTIVATIONS = ['relu', 'silu', 'tanh']
NUM_LAYERS = 3
INPUT_DIM = 512
mlp_bench = []

for combo in itertools.product(HIDDEN_SIZES, ACTIVATIONS, repeat=NUM_LAYERS):
    layers = []
    in_dim = INPUT_DIM
    for i in range(NUM_LAYERS):
        out_dim = combo[i*2]
        act = combo[i*2+1]
        layers.append({'in': in_dim, 'out': out_dim, 'activation': act})
        in_dim = out_dim

    # we cant encode it otherwise we won be able to extract data smoothly.
    # arch_key = encode_arch(layers)

    # Simulate fitness score or load from file
    # fitness = simulate_score(layers)  # or train_and_eval(...)
    mlp_bench.append(layers)

with open("mlp_bench.json", "a") as f:
    json.dump(mlp_bench, f)