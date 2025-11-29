# IO Libraries
import sys
import os
import pickle

# Standard Libraries
import numpy as np
import pandas as pd
import glob

# Torch Libraries
from torch.utils.data import DataLoader
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



  # your provided dataset wrapper
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle as sk_shuffle
from datetime import datetime
from pathlib import Path
import json


## Own Functions
from Models.AEmodels import VAECNN
from Models.datasets import Standard_Dataset
from utils import *
from config2 import *
torch.backends.cudnn.benchmark = True


def VAEConfigs(Config):
    inputmodule_paramsEnc = {'dim_input': 256, 'num_input_channels': 3}
    inputmodule_paramsDec = {'dim_input': 256}
    dim_in = inputmodule_paramsEnc['dim_input']

    net_paramsEnc = {}
    net_paramsDec = {}
    net_paramsRep = {}

    if Config == '1':
        net_paramsEnc['block_configs']=[[32, 32], [64, 64], [64, 64]]
        net_paramsEnc['stride']=[[1, 2], [1, 2], [1, 2]]
        net_paramsDec['block_configs']=[[64, 64], [64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsDec['block_configs'][0][0]

        net_paramsRep['h_dim']=16384
        net_paramsRep['z_dim']=256

    elif Config == '2':
        net_paramsEnc['block_configs']=[[32], [64], [128], [256]]
        net_paramsEnc['stride']=[[2], [2], [2], [2]]
        net_paramsDec['block_configs']=[[256], [128], [64], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsDec['block_configs'][0][0]

        net_paramsRep['h_dim']=16384
        net_paramsRep['z_dim']=512

    elif Config == '3':
        net_paramsEnc['block_configs']=[[32], [64], [64]]
        net_paramsEnc['stride']=[[1], [2], [2]]
        net_paramsDec['block_configs']=[[64], [32], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsDec['block_configs'][0][0]

        net_paramsRep['h_dim']=65536
        net_paramsRep['z_dim']=256

    return net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec, net_paramsRep


######################### 0. EXPERIMENT PARAMETERS
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

crossval_cropped_folders = get_all_subfolders(CROPPED_PATCHES_DIR)
annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)


print(f"Found {len(crossval_cropped_folders)} cropped folders and {len(annotated_folders)} annotated folders.")
# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3

# 0.1 NETWORK TRAINING PARAMS
VAE_params = {
    'epochs': 50,
    'batch_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'img_size': (128,128),
    'beta_start': 0.0,
    'beta_max': 1.0,
    'beta_warmup_epochs': 40,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# 0.2 FOLDERS

#### 1. LOAD DATA: Implement

# 1.1 Patient Diagnosis
vae_train_ims, vae_train_meta = LoadCropped(
    crossval_cropped_folders, n_images_per_folder=30,
    excelFile=PATIENT_DIAGNOSIS_FILE, resize=VAE_params['img_size']
)

print("Cropped loaded:", vae_train_ims.shape, vae_train_meta.shape)
print(vae_train_meta.head())

# Annotated para aprender umbral de error (ROC)
"""ann_ims, ann_meta = LoadAnnotated(
    annotated_folders, n_images_per_folder=5, excelFile=ANNOTATED_METADATA_FILE,
    resize=VAE_params['img_size']
)
print("Annotated loaded:", ann_ims.shape, ann_meta.shape)
print(ann_meta.head())
"""
#### 2. DATA SPLITING INTO INDEPENDENT SETS

# 2.0 Annotated set for FRed optimal threshold
# later 
# 2.1 AE trainnig set

# 2.1 Diagosis crossvalidation set

#### 3. lOAD PATCHES

#this function takes images and metadata and returns a Standard_Dataset object 
def _to_dataset(ims, meta, with_labels=False):
    X = np.stack([im.transpose(2, 0, 1) for im in ims], axis=0).astype(np.float32) / 255.0
    if with_labels:
        y = meta['Presence'].to_numpy(dtype=np.int64)
        return Standard_Dataset(X, y)
    else:
        return Standard_Dataset(X)

vae_train_ds = _to_dataset(vae_train_ims, vae_train_meta, with_labels=False)
vae_train_loader = DataLoader(vae_train_ds, batch_size=VAE_params['batch_size'],
                             shuffle=True)


#### 4. VAE TRAINING
Config = '1'

net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec, net_paramsRep = VAEConfigs(Config)

# Build VAE model (do not modify VAECNN implementation)
model = VAECNN(
    inputmodule_paramsEnc,
    net_paramsEnc,
    inputmodule_paramsDec,
    net_paramsDec,
    net_paramsRep
)

model.to(VAE_params['device'])

# 3. Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=VAE_params["lr"],
    weight_decay=VAE_params["weight_decay"]
)


def kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

loss_history = {
    "recon": [],
    "kld": [],
    "total": [],
}

# 4. Training loop
model.train()
for epoch in range(VAE_params["epochs"]):
    epoch_recon_loss = 0.0
    epoch_kld_loss   = 0.0
    epoch_total_loss = 0.0
    beta_step = (VAE_params["beta_max"] - VAE_params["beta_start"]) / max(1, VAE_params["beta_warmup_epochs"])
    beta_t = min(VAE_params["beta_max"], VAE_params["beta_start"] + epoch * beta_step)

    for batch in vae_train_loader:
        x = batch.to(torch.float32).to(VAE_params["device"])

        optimizer.zero_grad()

        # Forward pass through VAE
        recon, mu, logvar = model(x)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)

        # KLD
        kld_per_sample = kl_loss(mu, logvar)
        kld_loss = kld_per_sample.mean()

        # Total loss
        total_loss = recon_loss + beta_t * kld_loss

        # Backprop
        total_loss.backward()
        optimizer.step()

        # accumulate
        bs = x.size(0)
        epoch_recon_loss += recon_loss.item() * bs
        epoch_kld_loss   += kld_loss.item() * bs
        epoch_total_loss += total_loss.item() * bs

    # Normalize by dataset size
    N = len(vae_train_ds)
    epoch_recon_loss /= N
    epoch_kld_loss   /= N
    epoch_total_loss /= N

    loss_history["recon"].append(epoch_recon_loss)
    loss_history["kld"].append(epoch_kld_loss)
    loss_history["total"].append(epoch_total_loss)


    # Print epoch summary
    print(
        f"[VAE Config {Config}] Epoch {epoch+1}/{VAE_params['epochs']} | "
        f"Recon: {epoch_recon_loss:.6f} | "
        f"Total: {epoch_total_loss:.6f} | "
        f"KLD: {epoch_kld_loss:.6f} "
    )

# 5. Save model
Path('checkpoints').mkdir(exist_ok=True)

save_dict = {
    "state_dict": model.state_dict(),
    "config": {
        "VAE_params": VAE_params,
        "net_paramsEnc": net_paramsEnc,
        "net_paramsDec": net_paramsDec,
        "inputmodule_paramsEnc": inputmodule_paramsEnc,
        "inputmodule_paramsDec": inputmodule_paramsDec,
        "net_paramsRep": net_paramsRep,
        "config_id": Config
    },
    "epoch": VAE_params["epochs"],
    "loss_history": loss_history,      # store recon + KL per epoch
    "scaler_stats": None,              # if you later add normalization
    "train_meta": vae_train_meta,      # optional: metadata snapshot
}
torch.save(save_dict, "checkpoints/VAE_System1.pth")
print(f'Saved VAE config {Config} checkpoint to checkpoints/VAE_System1.pth')
# 6. Cleanup
del model, optimizer
torch.cuda.empty_cache()
gc.collect()
