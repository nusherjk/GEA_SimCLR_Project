import random
import copy
from functools import partial

ACTIVATIONS = ["relu", "silu", "tanh"]


def validate_config(config, input_dim=512, output_dim=128):
    # Ensure first layer matches encoder output
    config[0]['in'] = input_dim
    # Ensure last layer matches projection output
    config[-1]['out'] = output_dim

    # Fix all intermediate connections
    for i in range(len(config) - 1):
        config[i + 1]['in'] = config[i]['out']
    return config


def mutate_config(config):
    """
    Apply one random mutation to a given MLP config.
    Config format: list of dicts like
    [{'in': 128, 'out': 256, 'activation': 'relu', 'batchnorm': True}, ...]
    """
    new_config = copy.deepcopy(config)
    mutation = random.choice(["depth", "width", "activation", "regularization"])
    # print(new_config)
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
        if 'batchnorm' not in new_config[idx]:
            # If not present, add it and enable it
            new_config[idx]['batchnorm'] = True
        else:
            # Toggle
            new_config[idx]['batchnorm'] = not new_config[idx]['batchnorm']

    return new_config


# def create_new_generation(configs, num_children):
#     """Create new configs from parent configs by mutation."""
#     new_gen = []
#     for _ in range(num_children):
#         parent = random.choice(configs)
#         child_cfg = mutate_config(parent)
#         new_gen.append(child_cfg)
#     return new_gen


def create_new_generation(configs, num_children):
    new_gen = []
    for _ in range(num_children):
        child_cfg = mutate_config(configs)
        new_gen.append(child_cfg)
    return new_gen

def update_population(old_population, num_children):
    """Append new generation to the existing population."""
    new_gen = create_new_generation(old_population, num_children)
    return old_population + new_gen