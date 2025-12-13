# extract_latents.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from PIL import Image

from utils import *
from Models.AEmodels import AutoEncoderCNN, AEWithLatent
from config2 import (
    ANNOTATED_PATCHES_DIR,
    PATIENT_DIAGNOSIS_FILE,
    ANNOTATED_METADATA_FILE,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# LOAD AUTOENCODER
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# EXTRACT LATENTS + RECON ERRORS
# ------------------------------------------------------------
@torch.no_grad()
def extract_latents_and_errors(model, ae, images_np):
    # Convert HWC ? CHW
    images_np = np.stack([im.transpose(2, 0, 1) for im in images_np], axis=0) / 255.0
    images_tensor = torch.from_numpy(images_np).float()

    all_latents = []
    all_errors = []

    for i in tqdm(range(len(images_tensor)), desc="Extracting Latents"):
        img = images_tensor[i].unsqueeze(0).to(device)

        latent, _ = model(img)        # latent from encoder
        recon = ae(img)               # reconstruction

        latent_flat = latent.flatten().cpu().numpy()
        mse_error = torch.mean((img - recon) ** 2).item()

        all_latents.append(latent_flat)
        all_errors.append(mse_error)

    return np.array(all_latents), np.array(all_errors)


# ------------------------------------------------------------
# CREATE LABELS (true or pseudo)
# ------------------------------------------------------------
def create_labels(metadata, errors, threshold=None, use_pseudolabels=False):
    if use_pseudolabels:
        assert threshold is not None, "Need a threshold for pseudolabeling"
        labels = (errors > threshold).astype(int)
        print("Using pseudo-labels:", np.bincount(labels))
    else:
        labels = metadata["Presence"].astype(int).values
        print("Using true labels:", np.bincount(labels))
    return labels


# ------------------------------------------------------------
# MAIN EXTRACTION PIPELINE
# ------------------------------------------------------------
def extract_and_save_latents(autoencoder_path, threshold=None, use_pseudolabels=False):
    print("Loading annotated images + metadata...")
    annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)

    images_np, metadata = LoadAnnotated(
        annotated_folders,
        patient_excel=PATIENT_DIAGNOSIS_FILE,
        n_images_per_folder=None,
        excelFile=ANNOTATED_METADATA_FILE,
        verbose=True,
    )

    # 1. Load AE
    model, ae = load_autoencoder(autoencoder_path)

    # 2. Extract latents and errors
    latents, errors = extract_latents_and_errors(model, ae, images_np)

    # 3. Create labels
    labels = create_labels(metadata, errors, threshold, use_pseudolabels)

    # 4. Train/test split
    idx_train, idx_test = train_test_split(
        np.arange(len(latents)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    lat_train, lab_train = latents[idx_train], labels[idx_train]
    lat_test,  lab_test  = latents[idx_test],  labels[idx_test]

    # 5. Save latents
    out_dir = "data/latent_vectors"
    os.makedirs(out_dir, exist_ok=True)

    np.savez(os.path.join(out_dir, "train_latents_labels_no_cl_no_pca.npz"),
             latents=lat_train, labels=lab_train)

    np.savez(os.path.join(out_dir, "test_latents_labels_no_cl_no_pca.npz"),
             latents=lat_test, labels=lab_test)

    print("\nSaved latent vectors to", out_dir)
    print("Train:", lat_train.shape, "| Test:", lat_test.shape)


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
extract_and_save_latents(
    autoencoder_path="checkpoints/manual_removed/AE_Config1.pth",
    threshold=None,
    use_pseudolabels=False   # TRUE LABELS
)
