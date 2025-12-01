import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm
import random

from utils import *

from Models.AEmodels import AutoEncoderCNN, AEWithLatent     
from TripletLoss.triplet_loss import TripletLoss 
from TripletLoss.datasets import TripletDataset           
from Models.TripletModels import Embedder

from config import CROPPED_PATCHES_DIR, ANNOTATED_PATCHES_DIR, PATIENT_DIAGNOSIS_FILE, ANNOTATED_METADATA_FILE, RECON_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_triplet(
    autoencoder_path,
    images_np,
    threshold,
    batch_size=64,
    epochs=30,
    lr=1e-3
):
    """
    autoencoder_path: trained AE weights (.pt)
    images_tensor: tensor of shape (N, C, H, W)
    threshold: reconstruction-error threshold for pseudo-labels
    """

    
    if isinstance(images_np, np.ndarray):
        images_np = np.stack([im.transpose(2,0,1) for im in images_np], axis=0) / 255.0 
        print(images_np.shape)
        # Case 1: channels-last (N, H, W, C)
        if images_np.ndim == 4 and images_np.shape[-1] in [1, 3]:
            images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()
        # Case 2: already channels-first
        elif images_np.ndim == 4 and images_np.shape[1] in [1, 3]:
            images_tensor = torch.from_numpy(images_np).float()
        else:
            raise ValueError("images_np must be shape (N, H, W, C) or (N, C, H, W)")
    else:
        raise ValueError("images_np must be a NumPy array")

    inputmodule_paramsEnc = {'num_input_channels': 3}

    print("\nLoading Autoencoder...")
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs('1') # ? Change for other configurations!
    print(net_paramsEnc.keys())
    ae = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
    ae.load_state_dict(torch.load(autoencoder_path, map_location=device))
    model = AEWithLatent(ae).to(device)
    model.eval()

    # Extract latent vectors for all images
    print("Extracting latent vectors...")
    all_latents = []
    all_errors = []

    with torch.no_grad():
        for i in tqdm(range(len(images_tensor))):

            # (1, C, H, W)
            img = images_tensor[i].unsqueeze(0).to(device)
            latent, recon = model(img)
            recon = ae(img)
            latent_flat = latent.flatten().cpu().numpy()
            mse_error = torch.mean((img - recon) ** 2).item()

            all_latents.append(latent_flat)
            all_errors.append(mse_error)

    all_latents = np.array(all_latents)
    all_errors = np.array(all_errors)
    print(all_errors)

    # Generate pseudo-labels using the threshold
    print("\nApplying threshold...")
    labels = (all_errors > threshold).astype(int)
    print("Label distribution:", np.bincount(labels))

    # Triplet Dataset & Loader
    dataset = TripletDataset(all_latents, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create Embedder + Triplet Loss
    embedder = Embedder(input_dim=all_latents.shape[1]).to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(embedder.parameters(), lr=lr)

    # Training loop
    print("\nTraining triplet model...")
    for epoch in range(epochs):
        epoch_loss = 0
        for anchor, pos, neg in loader:
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            # get embeddings
            z_a = embedder(anchor)
            z_p = embedder(pos)
            z_n = embedder(neg)

            loss = criterion(z_a, z_p, z_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss = {epoch_loss / len(loader):.4f}")

    # 9. Save the embedder
    torch.save(embedder.state_dict(), "triplet_embedder.pt")
    print("\nSaved triplet embedder → triplet_embedder.pt")

    return embedder, all_latents, labels


df_diag = pd.read_csv(PATIENT_DIAGNOSIS_FILE) if os.path.isfile(PATIENT_DIAGNOSIS_FILE) else None

# 1.2 Patches Data
annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)

ae_val_ims, ae_val_meta = LoadAnnotated(
    annotated_folders, patient_excel=PATIENT_DIAGNOSIS_FILE,n_images_per_folder=1, 
    excelFile=ANNOTATED_METADATA_FILE, verbose=True
)


thresholds_df = pd.read_csv("../data/best_thresholds.csv")
best_threshold_mean_rgb = thresholds_df[thresholds_df['metric'] == 'mse_rgb']['best_threshold_mean'].item()
autoencoder_path = "checkpoints/AE_Config1.pth"

embedder, latents, labels = train_triplet(
    autoencoder_path,
    images_np=ae_val_ims,
    threshold=best_threshold_mean_rgb,
    batch_size=32, # for testing porpuses
    epochs=1, # for testing porpuses
    lr=1e-3
)

# TODO: Acabar de refctorizar el código, porque los paths harcodeados están mal y algunas clases las he puesto en utils y otros lados y no se si las he importado
# Tenetr en cuenta que el Chat me ha puest ocosas en cpu porque no tengo cuda, así que mirar de quetar el .cpu() a ver si funciona