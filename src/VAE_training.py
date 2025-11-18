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
from Models.AEmodels import VAECNN, Encoder
from Models.datasets import Standard_Dataset
from utils import *
from config import CROPPED_PATCHES_DIR, ANNOTATED_PATCHES_DIR, PATIENT_DIAGNOSIS_FILE, ANNOTATED_METADATA_FILE


def AEConfigs(Config):
    net_paramsEnc={}
    net_paramsDec={}
    inputmodule_paramsDec={}
    if Config=='1':
        # CONFIG1
        net_paramsEnc['block_configs']=[[32,32],[64,64]]
        net_paramsEnc['stride']=[[1,2],[1,2]]
        net_paramsDec['block_configs']=[[64,32],[32,inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
     

        
    elif Config=='2':
        # CONFIG 2
        net_paramsEnc['block_configs']=[[32],[64],[128],[256]]
        net_paramsEnc['stride']=[[2],[2],[2],[2]]
        net_paramsDec['block_configs']=[[128],[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
   
        
    elif Config=='3':  
        # CONFIG3
        net_paramsEnc['block_configs']=[[32],[64],[64]]
        net_paramsEnc['stride']=[[1],[2],[2]]
        net_paramsDec['block_configs']=[[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
    
    return net_paramsEnc,net_paramsDec,inputmodule_paramsDec


######################### 0. EXPERIMENT PARAMETERS


crossval_cropped_folders = get_all_subfolders(CROPPED_PATCHES_DIR)
annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)


print(f"Found {len(crossval_cropped_folders)} cropped folders and {len(annotated_folders)} annotated folders.")
# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3

# 0.1 NETWORK TRAINING PARAMS
VAE_params = {
    'epochs': 25,
    'batch_size': 256,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'img_size': (128,128),   
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# 0.2 FOLDERS

#### 1. LOAD DATA: Implement

# 1.1 Patient Diagnosis
crossval_cropped_folders = get_all_subfolders(CROPPED_PATCHES_DIR)

ae_train_ims, ae_train_meta = LoadCropped(
    crossval_cropped_folders, n_images_per_folder=3,
    excelFile=PATIENT_DIAGNOSIS_FILE, resize=VAE_params['img_size']
)

print("Cropped loaded:", ae_train_ims.shape, ae_train_meta.shape)
print(ae_train_meta.head())


# Annotated para aprender umbral de error (ROC)
"""ann_ims, ann_meta = LoadAnnotated(
    annotated_folders, n_images_per_folder=5, excelFile=ANNOTATED_METADATA_FILE,
    resize=VAE_params['img_size']
)
print("Annotated loaded:", ann_ims.shape, ann_meta.shape)
print(ann_meta.head())
"""

print(f"Found {len(crossval_cropped_folders)} cropped folders and {len(annotated_folders)} annotated folders.")
#### 2. DATA SPLITING INTO INDEPENDENT SETS

# 2.0 Annotated set for FRed optimal threshold
# later 
# 2.1 AE trainnig set

# 2.1 Diagosis crossvalidation set

#### 3. lOAD PATCHES

#this function takes images and metadata and returns a Standard_Dataset object 
def _to_dataset(ims, meta, with_labels=False):
    X = np.stack([im.transpose(2,0,1) for im in ims], axis=0) / 255.0
    if with_labels:
        y = np.array([m['Presence'] for m in meta], dtype=np.int64)
        return Standard_Dataset(X, y)
    else:
        return Standard_Dataset(X)

ae_train_ds = _to_dataset(ae_train_ims, ae_train_meta, with_labels=False)
ae_train_loader = DataLoader(ae_train_ds, batch_size=VAE_params['batch_size'], shuffle=True)

# ann_ds = _to_dataset(ann_ims, ann_meta, with_labels=True)

# ann_loader = DataLoader(ann_ds, batch_size=VAE_params['batch_size'], 
#                         shuffle=False, num_workers=2)

### 4. AE TRAINING

# EXPERIMENTAL DESIGN:
# TRAIN ON AE PATIENTS AN AUTOENCODER, USE THE ANNOTATED PATIENTS TO SET THE
# THRESHOLD ON FRED, VALIDATE FRED FOR DIAGNOSIS ON A 10 FOLD SCHEME OF REMAINING
# CASES.

# 4.1 Data Split

###### CONFIG1
Config='3'
net_paramsEnc, net_paramsDec, inputmodule_paramsDec=AEConfigs(Config)

tmp_encoder = Encoder(inputmodule_paramsEnc, net_paramsEnc)
tmp_encoder.eval()

with torch.no_grad():
    dummy = torch.zeros(
        1,
        inputmodule_paramsEnc['num_input_channels'],
        VAE_params['img_size'][0],
        VAE_params['img_size'][1],
    )
    h = tmp_encoder(dummy)          # (1, C', H', W')
    h_dim = h.view(1, -1).size(1)   # flatten → size h_dim

# define the parameters for the bottleneck representation
net_paramsRep = {
    'h_dim': h_dim,
    'z_dim': 16,  
}

model=VAECNN(inputmodule_paramsEnc, net_paramsEnc, 
             inputmodule_paramsDec, net_paramsDec,
             net_paramsRep).to(VAE_params['device'])

# 4.2 Model Training
optimizer = optim.Adam(model.parameters(), lr=VAE_params['lr'], weight_decay=VAE_params['weight_decay'])
criterion = nn.MSELoss(reduction='mean')

beta = 2.0
 
# KL for element and then the mean over batch 
def kl_loss(mu, logvar):
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(kld_per_sample)


def recon_loss_fn(x_recon, x):
    # Sumamos sobre todos los píxeles y canales, luego promedio por batch
    num_pixels = x.size(1) * x.size(2) * x.size(3)  # C*H*W
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / (x.size(0) * num_pixels)
    return recon_loss


model.train()
for epoch in range(VAE_params['epochs']):
    epoch_loss = 0.0 # total loss
    epoch_recon = 0.0 # reconstruction loss
    epoch_kl = 0.0 # KL divergence loss

    for batch in ae_train_loader:
        x = batch.to(VAE_params['device']).to(torch.float32)  
        #print("Shape after dataloader: "+str(x.shape))
        optimizer.zero_grad()

        x_recon, mu, logvar = model(x) # forward pass (edited for VAE)

        recon_loss = recon_loss_fn(x_recon, x)
        kl = kl_loss(mu, logvar)
        loss = recon_loss + beta * kl 

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        epoch_loss += loss.item() * batch_size
        epoch_recon += recon_loss.item() * batch_size
        epoch_kl += kl.item() * batch_size
        with torch.no_grad():
            print(f"mu mean={mu.mean().item():.4f}, mu std={mu.std().item():.4f}, "
                f"logvar mean={logvar.mean().item():.4f}, logvar std={logvar.std().item():.4f}")


    epoch_loss /= len(ae_train_ds)
    epoch_recon /= len(ae_train_ds)
    epoch_kl /= len(ae_train_ds)

    print(f"[VAE][Epoch {epoch+1}/{VAE_params['epochs']}] loss={epoch_loss:.5f} | recon={epoch_recon:.5f} | kld={epoch_kl:.5f}\n")


Path('checkpoints').mkdir(exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/VAE_System3.pth') # save model

# Free GPU Memory After Training
gc.collect()
torch.cuda.empty_cache()
