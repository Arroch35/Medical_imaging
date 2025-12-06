import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from Models.TripletModels import Embedder
from utils import *
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------
# Extract triplet embeddings from AE latent vectors
# ----------------------------------------------------------
def extract_new_latents(embedder, old_latents):
    embedder.eval()
    old_latents_t = torch.tensor(old_latents, dtype=torch.float32).to(device)

    new = []
    with torch.no_grad():
        for x in tqdm(old_latents_t, desc="Embedding with triplet model"):
            z = embedder(x.unsqueeze(0))               # (1,D)
            new.append(z.squeeze(0).cpu().numpy())     # (D,)
    return np.array(new)


# ----------------------------------------------------------
# Save arrays + CSV metadata
# ----------------------------------------------------------
def save_latents(np_array, labels, save_path_prefix):
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    np.save(save_path_prefix + "_latents.npy", np_array)
    np.save(save_path_prefix + "_labels.npy", labels)
    print(f"Saved → {save_path_prefix}_latents.npy and {save_path_prefix}_labels.npy")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def save_all_latents_from_saved(
    embedder_weights,
    latent_dir="../data/latent_vectors",
    output_dir="../data/latent_outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    # Load triplet embedder
    embedder = Embedder(input_dim=64*64*64).to(device)  
    embedder.load_state_dict(torch.load(embedder_weights, map_location=device))
    embedder.eval()

    # Load previously saved train latent vectors and labels
    train_data = np.load(os.path.join(latent_dir, "train_latents_labels.npz"))
    lat_train, y_train = train_data["latents"], train_data["labels"]

    # Load previously saved test latent vectors and labels
    test_data = np.load(os.path.join(latent_dir, "test_latents_labels.npz"))
    lat_test, y_test = test_data["latents"], test_data["labels"]

    # -------------------------
    # Compute new latent vectors
    # -------------------------
    print("\nComputing new train latent embeddings...")
    new_lat_train = extract_new_latents(embedder, lat_train)

    print("\nComputing new test latent embeddings...")
    new_lat_test = extract_new_latents(embedder, lat_test)

    # -------------------------
    # Save results
    # -------------------------
    save_latents(lat_train, y_train, os.path.join(output_dir, "train_old"))
    save_latents(new_lat_train, y_train, os.path.join(output_dir, "train_new"))

    save_latents(lat_test, y_test, os.path.join(output_dir, "test_old"))
    save_latents(new_lat_test, y_test, os.path.join(output_dir, "test_new"))

    print("\nDONE. Saved all train and test latent vectors and labels.")
    return lat_train, new_lat_train, y_train, lat_test, new_lat_test, y_test


# ----------------------------------------------------------
# Example run
# ----------------------------------------------------------
if __name__ == "__main__":
    save_all_latents_from_saved(
        embedder_weights="checkpoints/CL/triplet_embedder.pt",
        latent_dir="data/latent_vectors",
        output_dir="../data/latent_outputs"
    )