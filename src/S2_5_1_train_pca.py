import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib

from utils import *
from Models.AEmodels import AutoEncoderCNN, AEWithLatent
from Models.TripletModels import Embedder, EmbedderLarge
from TripletLoss.triplet_loss import TripletLoss
from TripletLoss.datasets import TripletDataset
from config import (
    ANNOTATED_PATCHES_DIR,
    PATIENT_DIAGNOSIS_FILE,
    ANNOTATED_METADATA_FILE,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------
# AUTOENCODER / LATENT EXTRACTION
# ------------------------------------------------------------------------------
def load_autoencoder(autoencoder_path):
    inputmodule_paramsEnc = {"num_input_channels": 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs("1", inputmodule_paramsEnc)

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
        print("Using pseudo-labels:", np.bincount(labels))
    else:
        labels = metadata["Presence"].astype(int).values
        print("Using true labels:", np.bincount(labels))
    return labels


# ------------------------------------------------------------------------------
# TRIPLET TRAINING (NO SEMI-HARD MINING)
# ------------------------------------------------------------------------------
def train_triplet_embedder(latents_train, labels_train, latents_test, labels_test,
                           batch_size=64, epochs=30, lr=1e-3, margin=0.3):

    train_dataset = TripletDataset(latents_train, labels_train)
    test_dataset = TripletDataset(latents_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    embedder = EmbedderLarge(input_dim=latents_train.shape[1]).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(embedder.parameters(), lr=lr)

    best_test_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):

        embedder.train()
        train_loss = 0

        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)

            za = embedder(a)
            zp = embedder(p)
            zn = embedder(n)

            loss = criterion(za, zp, zn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Evaluate ----
        embedder.eval()
        test_loss = 0

        with torch.no_grad():
            for a, p, n in test_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                za = embedder(a)
                zp = embedder(p)
                zn = embedder(n)
                test_loss += criterion(za, zp, zn).item()

        avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.4f} | Test: {avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_state = embedder.state_dict()

    # Save best model
    os.makedirs("checkpoints/CL", exist_ok=True)
    torch.save(best_state, "checkpoints/CL/embedder_triplet_best.pt")
    print("Saved → checkpoints/CL/embedder_triplet_best.pt")

    embedder.load_state_dict(best_state)
    return embedder


# ------------------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ------------------------------------------------------------------------------
def run_training(autoencoder_path, threshold, use_pseudolabels=True):

    print("Loading annotated images + metadata...")
    annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)

    images_np, metadata = LoadAnnotated(
        annotated_folders,
        patient_excel=PATIENT_DIAGNOSIS_FILE,
        n_images_per_folder=None,
        excelFile=ANNOTATED_METADATA_FILE,
        verbose=True,
    )

    # 1. Load autoencoder
    model, ae = load_autoencoder(autoencoder_path)

    # 2. Extract latent vectors + reconstruction errors
    latents, errors = extract_latents_and_errors(model, ae, images_np)

    # 3. Labels
    labels = create_labels(metadata, errors, threshold, use_pseudolabels)

    # 4. Train/Test split
    idx_train, idx_test = train_test_split(
        np.arange(len(latents)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    lat_train, lab_train = latents[idx_train], labels[idx_train]
    lat_test,  lab_test  = latents[idx_test],  labels[idx_test]

    # 5. PCA REDUCTION
    pca_dim = 2048
    pca = PCA(n_components=pca_dim, random_state=42)

    lat_train_pca = pca.fit_transform(lat_train)
    lat_test_pca  = pca.transform(lat_test)

    os.makedirs("checkpoints/CL", exist_ok=True)
    joblib.dump(pca, "checkpoints/CL/pca_reducer.joblib")
    print("Saved PCA → checkpoints/CL/pca_reducer.joblib")

    # 6. Train embedder with TripletDataset
    embedder = train_triplet_embedder(
        lat_train_pca, lab_train,
        lat_test_pca, lab_test,
        batch_size=128, epochs=30, lr=1e-3
    )

    return embedder, lat_train_pca, lab_train, lat_test_pca, lab_test



# ------------------------------------------------------------------------------
# RUN TRAINING + SAVE LATENTS
# ------------------------------------------------------------------------------
embedder, lat_train, y_train, lat_test, y_test = run_training(
    autoencoder_path="checkpoints/manual_removed/AE_Config1.pth",
    threshold=None,
    use_pseudolabels=False,   # TRUE LABELS
)

latent_dir = "../data/latent_vectors"
os.makedirs(latent_dir, exist_ok=True)

np.savez(os.path.join(latent_dir, "train_latents_labels.npz"),
         latents=lat_train, labels=y_train)

np.savez(os.path.join(latent_dir, "test_latents_labels.npz"),
         latents=lat_test, labels=y_test)

print("Saved latent vectors to", latent_dir)
