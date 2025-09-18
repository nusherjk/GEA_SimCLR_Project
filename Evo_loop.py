import random
import copy

ACTIVATIONS = ["relu", "silu", "tanh"]

def mutate_config(config):
    """
    Apply one random mutation to a given MLP config.
    Config format: list of dicts like
    [{'in': 128, 'out': 256, 'activation': 'relu', 'batchnorm': True}, ...]
    """
    new_config = copy.deepcopy(config)
    mutation = random.choice(["depth", "width", "activation", "regularization"])
    print(new_config)
    if mutation == "depth":
        # Add or remove a layer
        if random.random() < 0.5 and len(new_config) > 1:
            # Remove a random hidden layer (not input layer)
            idx = random.randrange(1, len(new_config))
            del new_config[idx]
        else:
            # Insert a new layer after some position
            insert_idx = random.randrange(len(new_config))
            in_dim = new_config[insert_idx]['out']
            out_dim = random.choice([64, 128, 256, 512])
            new_layer = {
                'in': in_dim,
                'out': out_dim,
                'activation': random.choice(ACTIVATIONS),
                'batchnorm': random.choice([True, False])
            }
            # update next layer input if it exists
            if insert_idx + 1 < len(new_config):
                new_config[insert_idx + 1]['in'] = out_dim
            new_config.insert(insert_idx + 1, new_layer)

    elif mutation == "width":
        # Change width of a random layer
        idx = random.randrange(len(new_config))
        delta = random.choice([-64, -32, 32, 64, 128])
        new_out = max(8, new_config[idx]['out'] + delta)
        new_config[idx]['out'] = new_out
        if idx + 1 < len(new_config):
            new_config[idx + 1]['in'] = new_out

    elif mutation == "activation":
        # Swap activation in a random layer
        idx = random.randrange(len(new_config))
        if 'activation' in new_config[idx]:
            current = new_config[idx]['activation']
            choices = [a for a in ACTIVATIONS if a != current]
            if choices:
                new_config[idx]['activation'] = random.choice(choices)

    elif mutation == "regularization":
        # Toggle batchnorm
        idx = random.randrange(len(new_config))
        new_config[idx]['batchnorm'] = not new_config[idx].get('batchnorm', False)

    return new_config


def create_new_generation(configs, num_children):
    """Create new configs from parent configs by mutation."""
    new_gen = []
    for _ in range(num_children):
        parent = random.choice(configs)
        child_cfg = mutate_config(parent)
        new_gen.append(child_cfg)
    return new_gen


def update_population(old_population, num_children):
    """Append new generation to the existing population."""
    new_gen = create_new_generation(old_population, num_children)
    return old_population + new_gen
import json
import numpy as np

# Example
if __name__ == "__main__":
    with open('utils/mlp_bench.json') as f:
        mlp_bench = json.load(f)

    random_configurations = list()
    lower_range = 0
    upper_range = len(mlp_bench) - 1

    positions = [np.random.randint(lower_range, upper_range + 1) for _ in range(2)]
    population  = list()
    for position in positions:
        population.append(mlp_bench[position])





    # evolve population for 2 generations
    for gen in range(2):
        print(f"\n=== Generation {gen} ===")
        population = update_population(population, num_children=3)  # add 3 new configs
        for i, cfg in enumerate(population):
            print(f"Config {i}:")
            print(cfg)
                # for layer in cfg:
                #     print("   ", layer)
