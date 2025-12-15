# Standard Libraries
import numpy as np
import pandas as pd
import glob
from PIL import Image
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Torch Libraries
from torch.utils.data import DataLoader
import gc
import torch
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid

import torch.nn as nn
import torch.optim as optim
import tqdm


# Own Functions
from Models.AEmodels import VAECNN
from Models.datasets import Standard_Dataset
from utils import *
from configVAE import *

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

        net_paramsRep['h_dim']=65536
        net_paramsRep['z_dim']=256

    elif Config == '2':
        net_paramsEnc['block_configs']=[[32], [64], [128], [256]]
        net_paramsEnc['stride']=[[2], [2], [2], [2]]
        net_paramsDec['block_configs']=[[256], [128], [64], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsDec['block_configs'][0][0]

        net_paramsRep['h_dim']=65536
        net_paramsRep['z_dim']=512

    elif Config == '3':
        net_paramsEnc['block_configs']=[[32], [64], [64]]
        net_paramsEnc['stride']=[[1], [2], [2]]
        net_paramsDec['block_configs']=[[64], [32], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsDec['block_configs'][0][0]

        net_paramsRep['h_dim']=262144
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
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'img_size': (256, 256),
    'beta_start': 0.0,
    'beta_max': 1.0,
    'beta_warmup_epochs': 40,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# 0.2 FOLDERS

#### 1. LOAD DATA: Implement

# 1.1 Patient Diagnosis
annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)

vae_val_ims, vae_val_meta = LoadAnnotated(
    annotated_folders, patient_excel=PATIENT_DIAGNOSIS_FILE,n_images_per_folder=None,
    excelFile=ANNOTATED_METADATA_FILE, resize=VAE_params['img_size'], verbose=True
)

print("Annotated loaded:", vae_val_ims.shape, vae_val_meta.shape)
print(vae_val_meta.head())


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

vae_val_ds = _to_dataset(vae_val_ims, vae_val_meta, with_labels=True) #Standard_Dataset object
vae_val_loader = DataLoader(vae_val_ds, batch_size=VAE_params['batch_size'],
                             shuffle=False)


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


# Load checkpoint --> trained weights
ckpt_path = f"checkpoints/VAE_Config{Config}.pth"
ckpt = torch.load(ckpt_path, map_location=device)

state_dict = ckpt["state_dict"]

model.load_state_dict(state_dict)       # because your .pth is just a state_dict
model.to(VAE_params['device'])
model.eval()


if not model:
    raise RuntimeError("No models loaded. Checkpoint files missing.")
print("Model rebuilt and weights loaded.")


# print head of the state dict
for key, value in state_dict.items():
    print(f"state dict= {key}: {value.shape}")
    break  # print only the first item

# ------------------ 3. Run reconstructions & save ------------------
global_idx = 0
total_samples = len(vae_val_ims)

pbar = tqdm.tqdm(total=total_samples, desc="Saving VAE reconstructions")


with torch.no_grad():
    for batch in vae_val_loader:
        if isinstance(batch, (list, tuple)):
            x_batch = batch[0]
        else:
            x_batch = batch

        x_batch = x_batch.to(device=device, dtype=torch.float32)
        batch_size = x_batch.shape[0]

        recon_batch, mu, logvar = model(x_batch)  #
        # detach and move to CPU because of no gradients
        # clamp is for valid image range
        recon_batch = recon_batch.detach().cpu().clamp(0.0, 1.0)

        # save each image in the batch
        for i in range(batch_size):
            idx = global_idx + i
            if idx >= len(vae_val_meta):
                continue

            row = vae_val_meta.iloc[idx]
            patid = str(row["PatID"])
            filename = str(row["imfilename"])

            out_dir = os.path.join(RECON_DIR, f"VAE_Config{Config}", patid)  # change name if needed
            os.makedirs(out_dir, exist_ok=True)

            # tensor to numpy array
            img_tensor = recon_batch[i]  # (C, H, W)
            # convert to (H, W, C) and scale to [0, 255]
            img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255.0).round().astype("uint8")

            out_path = os.path.join(out_dir, filename)
            try:
                # create PIL image from the numpy array in RGB format
                pil = Image.fromarray(img_np)
                pil.save(out_path)
            except Exception as e:
                print(f"[Save Error] idx {idx} file {out_path}: {e}")

        global_idx += batch_size
        pbar.update(batch_size)

    pbar.close()
    print("All VAE reconstructions saved to:", RECON_DIR)