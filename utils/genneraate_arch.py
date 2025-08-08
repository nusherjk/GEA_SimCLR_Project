import itertools
import json
import uuid

from MLP_bench import encode_arch
import torch
import numpy as np

# Generaate Arch generates the MLP benchmark configurations initially needed for creating random population

def get_batch_jacobian(net, x):
    torch.autograd.set_detect_anomaly(True)
    net.zero_grad()

    x.requires_grad_(True)

    _, y = net(x)
    like_y = torch.ones_like(y)
    y.backward(like_y)
    jacob = x.grad.detach()

    return jacob#, grad


def eval_score_perclass(jacob, labels=None, n_classes=10):
    k = 1e-5
    #n_classes = len(np.unique(labels))
    per_class={}
    for i, label in enumerate(labels):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label],jacob[i]))
        else:
            per_class[label] = jacob[i]

    ind_corr_matrix_score = {}
    for c in per_class.keys():
        s = 0
        try:
            corrs = np.corrcoef(per_class[c])

            s = np.sum(np.log(abs(corrs)+k))#/len(corrs)
            if n_classes > 100:
                s /= len(corrs)
        except: # defensive programming
            continue

        ind_corr_matrix_score[c] = s

    # per class-corr matrix A and B
    score = 0
    ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()
    if n_classes <= 100:

        for c in ind_corr_matrix_score_keys:
            # B)
            score = score + np.absolute(ind_corr_matrix_score[c])
    else:
        for c in ind_corr_matrix_score_keys:
            # A)
            for cj in ind_corr_matrix_score_keys:
                score = score + np.absolute(ind_corr_matrix_score[c]-ind_corr_matrix_score[cj])

        # should divide by number of classes seen
        score /= len(ind_corr_matrix_score_keys)

    return score



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