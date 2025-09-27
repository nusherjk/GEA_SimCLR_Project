import torch.cuda

from models.encoder_simCLR import Encoder
from models.mlp_projector import MLPProjector
from load_cifer10 import load_cifar10_f_jacobian

from Jacobian.Jacobian_score import load_cifar10_batch, get_batch_jacobian, eval_score_perclass
from Evolution.Evolution import update_population, create_new_generation, mutate_config
from train.SimClr_train import train_simclr
from train.eval_simCLR import evaluate
import json
import numpy as np

BATCH_SIZE =64
def construct_SimClr(mlp_config):
    encoder = Encoder()
    projection_head = MLPProjector(mlp_config)

    return encoder, projection_head


def calculateZeroProxy(config):
    encoder, projector = construct_SimClr(arch)
    # model = SimCLR(arch)

    x, y = load_cifar10_batch(BATCH_SIZE)

    y = y.cpu().tolist()
    jacobs_batch = get_batch_jacobian(encoder, projector, x)
    jacobs_batch = jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().tolist()
    try:
        score = eval_score_perclass(jacobs_batch, y)
    except Exception as e:
        print(e)
        score = np.nan
    return score



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    population_queue = list()
    history = list()
    C = 10

    P = 7

    # Initialize 10 MLP Heads randomly from MlP Bench which means C=10
    # train_loader = load_cifar10_f_jacobian(64)
    with open('utils/mlp_bench.json') as f:
        mlp_bench = json.load(f)

    '''
    random_configurations : Population Queue for
    '''
    random_configurations = list()
    lower_range = 0
    upper_range = len(mlp_bench) - 1

    positions = [np.random.randint(lower_range, upper_range + 1) for _ in range(C)]
    for position in positions:

        arch = mlp_bench[position]
        # encoder, projector = construct_SimClr(arch)
        # # model = SimCLR(arch)
        #
        # x, y = load_cifar10_batch(64)
        #
        # y = y.cpu().tolist()
        # jacobs_batch = get_batch_jacobian(encoder, projector, x)
        # jacobs_batch = jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().tolist()
        # try:
        #     score = eval_score_perclass(jacobs_batch, y)
        # except Exception as e:
        #     print(e)
        #     score = np.nan
        score = calculateZeroProxy(arch)
        random_configurations.append({
            # "position": position,
            "mlp_conf": arch,
            "score": score
        })

    # pprint.pprint(random_configurations)
    # Rank by highest score.
    sorted_random_configurations = sorted(random_configurations, key=lambda x: x["score"], reverse=True)

    # Select top 30%
    # Drop C -P worst individuals from population p=7
    selected_population = sorted_random_configurations[:3]

    # Train this selected population for 1 epochs and evaluate

    for architecture in selected_population:
        encoder, projector = train_simclr(root="./data",
                                        epochs=epochs,
                                        batch_size=BATCH_SIZE,
                                        lr=3e-4,
                                        weight_decay=1e-6,
                                        temperature=0.5,
                                        projector_config=architecture['mlp_conf'],  # use default
                                        device=device,
                                        num_workers=2,
                                          )

        top_1_accuracy = evaluate(encoder, device)
        print(top_1_accuracy)
        history.append({
            # "position": architecture['position'],
            "mlp_conf": architecture['mlp_conf'],
            "accuracy": top_1_accuracy
        })

    # Rank by highest accuracy.
    history = sorted(history, key=lambda x: x["accuracy"], reverse=True)



    while len(history) < C:
        parent = history[0] # take the best accuracy one.
        generation = list()
        while len(generation)<P:
            child_arch = mutate_config(parent['mlp_head'])
            child_arch_score = calculateZeroProxy(child_arch)
            generation.append({
                "mlp_conf": child_arch,
                "score": child_arch_score
            })

        generation = sorted(generation, key=lambda x: x["score"], reverse=True)
        top_child = generation[0]
        encoder, projector = train_simclr(root="./data",
                                          epochs=1,
                                          batch_size=128,
                                          lr=3e-4,
                                          weight_decay=1e-6,
                                          temperature=0.5,
                                          projector_config=top_child['mlp_conf'],  # use default
                                          device=device,
                                          num_workers=2,
                                          )


        top_child_accuracy = evaluate(encoder, device)
        history.append({
            "mlp_conf": top_child,
            "accuracy": top_child_accuracy
        })







    # pprint.pprint(selected_population)

    # evolve population for 2 generations
    # for gen in range(2):
    #
    #     print(f"\n=== Generation {gen} ===")
    #     new_gens = list()
    #     # population_queue = update_population(selected_population, num_children=3)
    #     for population in selected_population:
    #         print(population['mlp_conf'])
    #         new_gens += create_new_generation(population['mlp_conf'],
    #                                       num_children=3)  # add 3 new configs for each population
    #         # new_gens += mutate_config(population['mlp_conf'])  # add 3 new configs for each population
    #
    #     for i, cfg in enumerate(new_gens):
    #         print(f"Config {i}:")
    #         for layer in cfg:
    #             print("   ", layer)
    #
    # print('Hello World!')