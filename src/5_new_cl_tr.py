import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import *
from Models.AEmodels import AutoEncoderCNN, AEWithLatent
from TripletLoss.triplet_loss import TripletLoss
from TripletLoss.datasets import TripletDataset
from Models.TripletModels import Embedder

from config import (
    ANNOTATED_PATCHES_DIR,
    PATIENT_DIAGNOSIS_FILE,
    ANNOTATED_METADATA_FILE,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_autoencoder(autoencoder_path):
    inputmodule_paramsEnc = {"num_input_channels": 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs("1")

    ae = AutoEncoderCNN(
        inputmodule_paramsEnc,
        net_paramsEnc,
        inputmodule_paramsDec,
        net_paramsDec,
    )
    ae.load_state_dict(torch.load(autoencoder_path, map_location=device))
    model = AEWithLatent(ae).to(device)
    model.eval()

    return model, ae

def extract_latents_and_errors(model, ae, images_np):
    # Convert (N,H,W,C) → tensor (N,C,H,W)
    images_np = np.stack([im.transpose(2, 0, 1) for im in images_np], axis=0) / 255.0
    images_tensor = torch.from_numpy(images_np).float()

    all_latents = []
    all_errors = []

    with torch.no_grad():
        for i in tqdm(range(len(images_tensor)), desc="Extracting Latents"):
            img = images_tensor[i].unsqueeze(0).to(device)

            latent, _ = model(img)
            recon = ae(img)

            latent_flat = latent.flatten().cpu().numpy()
            mse_error = torch.mean((img - recon) ** 2).item()

            all_latents.append(latent_flat)
            all_errors.append(mse_error)

    return np.array(all_latents), np.array(all_errors)

def create_labels(metadata, errors, threshold, use_pseudolabels):
    if use_pseudolabels:
        labels = (errors > threshold).astype(int)
        print("Using pseudo-labels. Distribution:", np.bincount(labels))
    else:
        labels = metadata["Presence"].astype(int).values
        print("Using TRUE labels. Distribution:", np.bincount(labels))

    return labels

def train_triplet_embedder(latents, labels, batch_size=64, epochs=30, lr=1e-3):
    dataset = TripletDataset(latents, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embedder = Embedder(input_dim=latents.shape[1]).to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(embedder.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        for anchor, pos, neg in loader:
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            z_a = embedder(anchor)
            z_p = embedder(pos)
            z_n = embedder(neg)

            loss = criterion(z_a, z_p, z_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")

    os.makedirs("/checkpoints/CL", exist_ok=True)
    torch.save(embedder.state_dict(), "/checkpoints/CL/triplet_embedder_true_labels.pt")
    print("Saved triplet embedder → triplet_embedder.pt")

    return embedder

def run_training(autoencoder_path, threshold, use_pseudolabels=True):

    print("Loading annotated images + metadata...")
    annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)

    images_np, metadata = LoadAnnotated(
        annotated_folders,
        patient_excel=PATIENT_DIAGNOSIS_FILE,
        n_images_per_folder=None,
        excelFile=ANNOTATED_METADATA_FILE,
        verbose=True,
    ) # ? Aquí estoy asuminedo que las imagenes y los labels están alineados, y teóricamente lo están, pero si hay algún error mirar primero aquí

    # 1. Load autoencoder
    model, ae = load_autoencoder(autoencoder_path)

    # 2. Extract latent vectors + reconstruction errors
    latents, errors = extract_latents_and_errors(model, ae, images_np)

    # 3. Select labels depending on the chosen mode
    labels = create_labels(metadata, errors, threshold, use_pseudolabels)

    # 4. Stratified split (works for both real and pseudo-labels)
    idx_train, idx_test = train_test_split(
        np.arange(len(latents)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    lat_train, y_train = latents[idx_train], labels[idx_train]
    lat_test,  y_test  = latents[idx_test],  labels[idx_test]

    print("Train distribution:", np.bincount(y_train))
    print("Test distribution:", np.bincount(y_test))

    # 5. Train triplet embedder on TRAIN only
    embedder = train_triplet_embedder(
        lat_train, y_train,
        batch_size=32,
        epochs=10,
        lr=1e-3
    )

    return embedder, lat_train, y_train, lat_test, y_test

embedder, lat_train, y_train, lat_test, y_test = run_training(
    autoencoder_path="checkpoints/AE_Config1.pth",
    threshold=None,                 # ignored
    use_pseudolabels=False          # REAL LABELS
)

latent_dir="data/latent_vectors"
os.makedirs(latent_dir, exist_ok=True)
# Save training latents and labels
np.savez(
    os.path.join(latent_dir, "train_latents_labels.npz"),
    latents=lat_train,
    labels=y_train
)

# Save test latents and labels
np.savez(
    os.path.join(latent_dir, "test_latents_labels.npz"),
    latents=lat_test,
    labels=y_test
)

print(f"Saved latent vectors and labels to {latent_dir}")

# thresholds_df = pd.read_csv("../data/best_thresholds.csv")
# best_threshold_mean_rgb = thresholds_df[thresholds_df['metric'] == 'mse_rgb']['best_threshold_mean'].item()

# embedder, lat_train, y_train, lat_test, y_test = run_training(
#     autoencoder_path="checkpoints/AE_Config1.pth",
#     threshold=best_threshold_mean_rgb,  
#     use_pseudolabels=True           # PSEUDO-LABELS
# )
