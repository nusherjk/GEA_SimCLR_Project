import torch
from Evolution.Evolution import mutate_config, validate_config
from fileManager import *
from main import calculateZeroProxy
from models.encoder_simCLR import Encoder
from models.mlp_projector import MLPProjector
from postTrain import post_train_linearClassifier
from train.SimClr_train import train_simclr
from train.eval_simCLR import evaluate
from torch.utils.tensorboard import SummaryWriter

import numpy as np
if __name__ == '__main__':
    # history = load_history()
    # selected_population = load_population()
    # print(len(history))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 1024
    EPOCHS = 800


    # ckpt_path = "results/simclr_arch_31.pt"
    # checkpoint = torch.load(ckpt_path, map_location=torch.device(DEVICE))  # or 'cuda'
    config = [{'in': 512, 'out': 192, 'activation': 'tanh'}, {'in': 192, 'out': 128, 'activation': 'tanh', 'batchnorm': True}, {'in': 128, 'out': 128, 'activation': 'silu'}]
    # Create your model instances first
    encoder, projector = train_simclr(root="./data",
                                      epochs=1,
                                      batch_size=BATCH_SIZE,
                                      lr=3e-4,
                                      weight_decay=1e-6,
                                      temperature=0.5,
                                      projector_config=config,  # use default
                                      device=DEVICE,
                                      num_workers=16,
                                      )


    accuracy = post_train_linearClassifier(encoder, DEVICE)


    print('top1 accuracy: {}'.format(accuracy))


    #
    # # Load model weights
    # encoder.load_state_dict(checkpoint["encoder_state_dict"])
    # projector.load_state_dict(checkpoint["projector_state_dict"])
    #
    # # Optionally restore epoch
    # start_epoch = checkpoint.get("epoch", 0)
    #
    # print(f"Checkpoint loaded successfully (epoch {start_epoch})")

    #
    #
    # i = len(history) - P
    #
    # while len(history) < C:
    #     i += 1
    #
    #     # take sample set len=S from the population
    #     sample = [selected_population[np.random.randint(0, len(selected_population))] for _ in range(P)]
    #     sample = sorted(sample, key=lambda x: x["score"], reverse=True)
    #     parent = sample[0]  # take the best scored one.
    #
    #     generation = list()
    #     while len(generation) < P:
    #         child_arch = mutate_config(parent['mlp_conf'])
    #         # print(child_arch)
    #         child_arch = validate_config(child_arch)
    #         # print(child_arch)
    #         child_arch_score = calculateZeroProxy(child_arch)
    #         generation.append({
    #             "mlp_conf": child_arch,
    #             "score": child_arch_score
    #         })
    #
    #     generation = sorted(generation, key=lambda x: x["score"], reverse=True)
    #     top_child = generation[0]
    #     encoder, projector = train_simclr(root="./data",
    #                                       epochs=EPOCHS,
    #                                       batch_size=BATCH_SIZE,
    #                                       lr=3e-4,
    #                                       weight_decay=1e-6,
    #                                       temperature=0.5,
    #                                       projector_config=top_child['mlp_conf'],  # use default
    #                                       device=DEVICE,
    #                                       num_workers=2,
    #                                       )
    #
    #     # torch.save({
    #     #     "encoder": encoder.state_dict(),
    #     #     "projector": projector.state_dict(),
    #     #     "config": top_child['mlp_conf']
    #     # }, f"checkpoints/simclr_arch_{i}.pt")
    #
    #     selected_population.append(top_child)
    #     top_child_accuracy = evaluate(encoder, DEVICE)
    #     # torch.save(encoder.state_dict(), "simclr_model.pth")
    #     writer.add_scalar("Generation/Accuracy", top_child_accuracy, i)
    #     ckpt_path = save_checkpoint(EPOCHS, encoder, projector, "gen_" + str(i))
    #     history.append({
    #         "encoder": str(encoder),
    #         "projector": str(projector),
    #         "mlp_conf": top_child["mlp_conf"],
    #         "accuracy": top_child_accuracy,
    #         "checkpoint": ckpt_path
    #     })
    #     selected_population.pop(0)
    #     try:
    #         save_history(history)
    #         print("history Updated!")
    #     except Exception:
    #         print("History Not updated!")
    #
    #     try:
    #         save_population(selected_population)
    #         print("Population Updated!")
    #     except Exception:
    #         print("Population Not updated!")
    #
    # sorted_history = sorted(history, key=lambda x: x["accuracy"])
    # top_performer = sorted_history[0]
    # # torch.save(top_performer['encoder'].state_dict(), "out/simclr_encoder.pth")
    # # torch.save(top_performer['projector'].state_dict(), "out/simclr_projector.pth")
    # writer.close()
    # print(f"best performing MLP configuration:{top_performer['mlp_conf']}")
