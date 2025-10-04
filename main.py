import pprint
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import sys
from models.encoder_simCLR import Encoder
from models.mlp_projector import MLPProjector

from Jacobian.Jacobian_score import load_cifar10_batch, get_batch_jacobian, eval_score_perclass
from Evolution.Evolution import mutate_config, validate_config
# from fileManager import *
from train.SimClr_train import train_simclr
from train.eval_simCLR import evaluate
import json
import numpy as np

# P/S/C=10/5/200.

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
EPOCHS = 25
C = 50  # 200
P = 10  # 10
S = 5 # 5

HISTORY_FILE = './history.json'
CHECKPOINT_DIR = "./results/checkpoints"
POPULATION_FILE = "./population.json"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# os.makedirs(HISTORY_FILE, exist_ok=True)
# os.makedirs(POPULATION_FILE, exist_ok=True)
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)   # print to console
        self.log.write(message)        # write to file

    def flush(self):
        pass  # needed for Python compatibility

sys.stdout = Logger("output_log.txt")


# --- History management ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# --- Checkpoint management ---
def save_checkpoint(epoch, encoder, projector, name):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "projector_state_dict": projector.state_dict(),
    }, ckpt_path)
    return ckpt_path

def load_checkpoint(path, encoder, projector, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    projector.load_state_dict(checkpoint["projector_state_dict"])
    return checkpoint["epoch"]





def load_population():
    if os.path.exists(HISTORY_FILE):
        with open(POPULATION_FILE, "r") as f:
            return json.load(f)
    return []


def save_population(population):
    with open(POPULATION_FILE, "w") as f:
        json.dump(population, f, indent=4)

########################################
def construct_SimClr(mlp_config):
    encoder = Encoder()
    projection_head = MLPProjector(mlp_config)

    return encoder, projection_head


def calculateZeroProxy(config):
    # writer = SummaryWriter(log_dir=f"./logs/models/{uuid.uuid4()}")
    encoder, projector = construct_SimClr(config)
    # model = SimCLR(arch)
    x, y = load_cifar10_batch(BATCH_SIZE)
    # writer.add_graph(model, x)
    # writer.close()
    y = y.cpu().tolist()
    jacobs_batch = get_batch_jacobian(encoder, projector, x)
    jacobs_batch = jacobs_batch.reshape(jacobs_batch.size(0), -1).to(DEVICE).tolist()
    try:
        score = eval_score_perclass(jacobs_batch, y)
    except Exception as e:
        print(e)
        score = np.nan
    return score


if __name__ == '__main__':
    writer = SummaryWriter(log_dir='./logs/GenerationsVAccuracy')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = EPOCHS
    population_queue = list()
    history = list()

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

        score = calculateZeroProxy(arch)
        random_configurations.append({
            "mlp_conf": arch,
            "score": score
        })

    pprint.pprint(random_configurations)
    # Rank by highest score.
    sorted_random_configurations = sorted(random_configurations, key=lambda x: x["score"], reverse=True)

    # Select top 30%
    # Drop C -P worst individuals from population
    selected_population = sorted_random_configurations[:P]
    save_population(selected_population)
    print("Population Updated!")

    # Train this selected population for Set epochs and evaluate

    for i, architecture in enumerate(selected_population):
        encoder, projector = train_simclr(root="./data",
                                          epochs=EPOCHS,
                                          batch_size=BATCH_SIZE,
                                          lr=3e-4,
                                          weight_decay=1e-6,
                                          temperature=0.5,
                                          projector_config=architecture['mlp_conf'],  # use default
                                          device=DEVICE,
                                          num_workers=2,
                                          )

        top_1_accuracy = evaluate(encoder, DEVICE)
        print(top_1_accuracy)
        ckpt_path = save_checkpoint(EPOCHS, encoder, projector, 'init_' + str(i))
        history.append({
            "encoder": str(encoder),
            "projector": str(projector),
            # "position": architecture['position'],
            "mlp_conf": architecture['mlp_conf'],
            "accuracy": top_1_accuracy,
            "checkpoint": ckpt_path
        })
        save_history(history)
        print("history Updated!")

    # Rank by highest accuracy.
    # history = sorted(history, key=lambda x: x["accuracy"], reverse=True)

    i = 0

    while len(history) < C:
        i += 1

        # take sample set len=S from the population
        sample = [selected_population[np.random.randint(0, len(selected_population))] for _ in range(P)]
        sample = sorted(sample, key=lambda x: x["score"], reverse=True)
        parent = sample[0]  # take the best scored one.

        generation = list()
        while len(generation) < P:
            child_arch = mutate_config(parent['mlp_conf'])
            # print(child_arch)
            child_arch = validate_config(child_arch)
            # print(child_arch)
            child_arch_score = calculateZeroProxy(child_arch)
            generation.append({
                "mlp_conf": child_arch,
                "score": child_arch_score
            })

        generation = sorted(generation, key=lambda x: x["score"], reverse=True)
        top_child = generation[0]
        encoder, projector = train_simclr(root="./data",
                                          epochs=EPOCHS,
                                          batch_size=BATCH_SIZE,
                                          lr=3e-4,
                                          weight_decay=1e-6,
                                          temperature=0.5,
                                          projector_config=top_child['mlp_conf'],  # use default
                                          device=DEVICE,
                                          num_workers=2,
                                          )

        # torch.save({
        #     "encoder": encoder.state_dict(),
        #     "projector": projector.state_dict(),
        #     "config": top_child['mlp_conf']
        # }, f"checkpoints/simclr_arch_{i}.pt")

        selected_population.append(top_child)
        top_child_accuracy = evaluate(encoder, DEVICE)
        # torch.save(encoder.state_dict(), "simclr_model.pth")
        writer.add_scalar("Generation/Accuracy", top_child_accuracy, i)
        ckpt_path = save_checkpoint(EPOCHS, encoder, projector, "gen_" + str(i))
        history.append({
            "encoder": str(encoder),
            "projector": str(projector),
            "mlp_conf": top_child["mlp_conf"],
            "accuracy": top_child_accuracy,
            "checkpoint": ckpt_path
        })
        selected_population.pop(0)
        try:
            save_history(history)
            print("history Updated!")
        except Exception:
            print("History Not updated!")

        try:
            save_population(selected_population)
            print("Population Updated!")
        except Exception:
            print("Population Not updated!")

    sorted_history = sorted(history, key=lambda x: x["accuracy"])
    top_performer = sorted_history[0]
    torch.save(top_performer['encoder'].state_dict(), "out/simclr_encoder.pth")
    torch.save(top_performer['projector'].state_dict(), "out/simclr_projector.pth")
    writer.close()
    print(f"best performing MLP configuration:{top_performer['mlp_conf']}")

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
