
import json
import os
import torch
HISTORY_FILE = 'history.json'
CHECKPOINT_DIR = "./results/checkpoints"
POPULATION_FILE = "population.json"



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
    if os.path.exists(POPULATION_FILE):
        with open(POPULATION_FILE, "r") as f:
            return json.load(f)
    return []


def save_population(population):
    with open(POPULATION_FILE, "w") as f:
        json.dump(population, f, indent=4)



#
#
# # --- Example usage ---
# history = load_history()
#
# for epoch in range(5):  # Replace with your evo loop
#     # Dummy data (replace with real encoder/projector/models)
#     encoder = torch.nn.Linear(128, 64)
#     projector = torch.nn.Linear(64, 32)
#     top_child = {"mlp_conf": {"hidden": [128, 64]}}
#     top_child_accuracy = 0.8 + epoch * 0.01
#
#     # Save checkpoint
#     ckpt_path = save_checkpoint(epoch, encoder, projector)
#
#     # Log history with checkpoint path
#     history.append({
#         "epoch": epoch,
#         "encoder": str(encoder),  # (or class name only)
#         "projector": str(projector),
#         "mlp_conf": top_child["mlp_conf"],
#         "accuracy": top_child_accuracy,
#         "checkpoint": ckpt_path
#     })
#
#     save_history(history)
#     print(f"Epoch {epoch}: saved checkpoint {ckpt_path}")
#
# # Reload history anytime
# reloaded = load_history()
# print("Reloaded:", reloaded)