import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *
from Models.AEmodels import AutoEncoderCNN, AEWithLatent
from TripletLoss.triplet_loss import TripletLoss
from TripletLoss.datasets import TripletDataset
from Models.TripletModels import Embedder, EmbedderLarge
from sklearn.decomposition import PCA
from config2 import (
    ANNOTATED_PATCHES_DIR,
    PATIENT_DIAGNOSIS_FILE,
    ANNOTATED_METADATA_FILE,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
# Semi-hard triplet mining function
def semi_hard_triplets(embeddings, labels, margin=1.0):
    """
    embeddings: (B,D)
    labels: (B,)
    Returns: anchor, positive, negative tensors
    """
    B = embeddings.size(0)
    anchors, positives, negatives = [], [], []

    for i in range(B):
        anchor = embeddings[i]
        label = labels[i]
        mask_pos = (labels == label) & (torch.arange(B, device=labels.device) != i)
        mask_neg = labels != label

        if mask_pos.sum() == 0 or mask_neg.sum() == 0:
            continue

        pos_dist = torch.norm(anchor - embeddings[mask_pos], dim=1)
        neg_dist = torch.norm(anchor - embeddings[mask_neg], dim=1)

        # Semi-hard: negative further than positive but within margin
        pos_idx = pos_dist.argmin()
        neg_candidates = (neg_dist > pos_dist[pos_idx]).nonzero(as_tuple=True)[0]
        if len(neg_candidates) == 0:
            continue
        neg_idx = neg_candidates[0]

        anchors.append(anchor)
        positives.append(embeddings[mask_pos][pos_idx])
        negatives.append(embeddings[mask_neg][neg_idx])

    if len(anchors) == 0:
        return None, None, None
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
    
    
def train_embedder(lat_train, y_train, lat_test, y_test, batch_size=256, epochs=30, lr=1e-3, margin=1.0):
    X_train = torch.from_numpy(lat_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_test = torch.from_numpy(lat_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)

    dataset = TensorDataset(X_train, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embedder = EmbedderLarge(input_dim=lat_train.shape[1]).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(embedder.parameters(), lr=lr)

    best_test_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, epochs+1):
        embedder.train()
        train_losses = []

        for batch_x, batch_y in loader:
            z = embedder(batch_x)
            z = nn.functional.normalize(z, p=2, dim=1)

            a, p, n = semi_hard_triplets(z, batch_y, margin=margin)
            if a is None:
                continue

            loss = criterion(a, p, n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0

        # Evaluate test loss
        embedder.eval()
        with torch.no_grad():
            z_test = embedder(X_test)
            z_test = nn.functional.normalize(z_test, p=2, dim=1)
            a_t, p_t, n_t = semi_hard_triplets(z_test, y_test_t, margin=margin)
            if a_t is not None:
                test_loss = criterion(a_t, p_t, n_t).item()
            else:
                test_loss = float("inf")

        print(f"Epoch {epoch}/{epochs} | Train loss: {avg_train_loss:.4f} | Test loss: {test_loss:.4f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state_dict = embedder.state_dict()

    # Save the best model only
    os.makedirs("checkpoints/CL", exist_ok=True)
    torch.save(best_state_dict, "checkpoints/CL/embedder_semi_hard_best.pt")
    print(f"Saved best embedder with test loss {best_test_loss:.4f} → checkpoints/CL/embedder_semi_hard_best.pt")

    return embedder


def train_triplet_embedder(latents, labels, latents_test, labels_test, batch_size=64, epochs=30, lr=1e-3):
    dataset = TripletDataset(latents, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TripletDataset(latents_test, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    embedder = Embedder(input_dim=latents.shape[1]).to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(embedder.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        embedder.train()
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

        avg_train_loss = epoch_loss / len(loader)

        # ---- Compute test loss ----
        embedder.eval()
        test_loss = 0
        with torch.no_grad():
            for anchor, pos, neg in test_loader:
                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                z_a = embedder(anchor)
                z_p = embedder(pos)
                z_n = embedder(neg)

                loss = criterion(z_a, z_p, z_n)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    os.makedirs("/checkpoints/CL", exist_ok=True)
    torch.save(embedder.state_dict(), "/checkpoints/CL/triplet_embedder_true_labels.pt")
    print("Saved triplet embedder → triplet_embedder_true_labels.pt")
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
    )

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
    lat_test, y_test = latents[idx_test], labels[idx_test]
    
    pca_dim = 2048  # or 128 or 512
    pca = PCA(n_components=pca_dim, random_state=42)
    
        # Save PCA model
    os.makedirs("checkpoints/CL", exist_ok=True)
    joblib.dump(pca, "checkpoints/CL/pca_reducer.joblib")
    print("Saved PCA → checkpoints/CL/pca_reducer.joblib")
    
    lat_train_pca = pca.fit_transform(lat_train)
    lat_test_pca  = pca.transform(lat_test)

    print("Train distribution:", np.bincount(y_train))
    print("Test distribution:", np.bincount(y_test))

    # 5. Train triplet embedder on TRAIN only
    embedder = train_embedder(
        lat_train_pca, y_train, lat_test_pca, y_test, 
        batch_size=256, epochs=30, lr=1e-3
    )

    return embedder, lat_train_pca, y_train, lat_test_pca, y_test

embedder, lat_train, y_train, lat_test, y_test = run_training(
    autoencoder_path="checkpoints/manual_removed/AE_Config1.pth",
    threshold=None,  # ignored
    use_pseudolabels=False  # REAL LABELS
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
#     use_pseudolabels=True  # PSEUDO-LABELS
# )
