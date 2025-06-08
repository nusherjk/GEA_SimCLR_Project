"""
Example: All 3-layer MLPs with:

Hidden sizes ∈ {64, 128, 256, 512}

Activations ∈ {ReLU, SiLU, Tanh}

This gives  ~1000–2000 unique architectures.


"""
import json


def encode_arch(layers):
    return str([(layer['out'], layer['activation']) for layer in layers])
############################


##################

with open("mlp_bench.json") as f:
    mlp_bench = json.load(f)

def evaluate_arch(arch):
    key = encode_arch(arch)
    return mlp_bench.get(key, {"score": 0})["score"]

###########
import random

def generate_random_architecture():
    layers = []
    in_dim = INPUT_DIM
    for _ in range(NUM_LAYERS):
        out_dim = random.choice(HIDDEN_SIZES)
        activation = random.choice(ACTIVATIONS)
        layers.append({'in': in_dim, 'out': out_dim, 'activation': activation})
        in_dim = out_dim
    return layers

population = [generate_random_architecture() for _ in range(pop_size)]
