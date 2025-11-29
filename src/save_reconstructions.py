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
import torch.nn as nn
import torch.optim as optim

# Own Functions
from Models.AEmodels import VAECNN, Encoder
from Models.datasets import Standard_Dataset

from config2 import CROPPED_PATCHES_DIR, ANNOTATED_PATCHES_DIR, PATIENT_DIAGNOSIS_FILE, ANNOTATED_METADATA_FILE, RECON_DIR
from utils import *
from tqdm import tqdm
# import wandb
import os


def AEConfigs(Config):
    net_paramsEnc = {}
    net_paramsDec = {}
    inputmodule_paramsDec = {}
    if Config == '1':
        # CONFIG1
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]



    elif Config == '2':
        # CONFIG 2
        net_paramsEnc['block_configs'] = [[32], [64], [128], [256]]
        net_paramsEnc['stride'] = [[2], [2], [2], [2]]
        net_paramsDec['block_configs'] = [[128], [64], [32], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]


    elif Config == '3':
        # CONFIG3
        net_paramsEnc['block_configs'] = [[32], [64], [64]]
        net_paramsEnc['stride'] = [[1], [2], [2]]
        net_paramsDec['block_configs'] = [[64], [32], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec


# 0.1 AE PARAMETERS
inputmodule_paramsEnc = {}
inputmodule_paramsEnc['num_input_channels'] = 3

# 0.1 NETWORK TRAINING PARAMS
VAE_params = {
    'epochs': 20,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'img_size': (256,256),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

#### 1. LOAD DATA: Implement
# 1.1 Patient Diagnosis
df_diag = pd.read_csv(PATIENT_DIAGNOSIS_FILE) if os.path.isfile(PATIENT_DIAGNOSIS_FILE) else None

# 1.2 Patches Data
annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)

vae_val_ims, vae_val_meta = LoadAnnotated(
    annotated_folders, patient_excel=PATIENT_DIAGNOSIS_FILE, n_images_per_folder=None,
    excelFile=ANNOTATED_METADATA_FILE, resize=VAE_params['img_size'], verbose=True
)


#### 3. lOAD PATCHES
def _to_dataset(ims, meta, with_labels=False):
    X = np.stack([im.transpose(2, 0, 1) for im in ims], axis=0) / 255.0
    if with_labels:
        y = meta['Presence'].to_numpy(dtype=np.int64)
        return Standard_Dataset(X, y)
    else:
        return Standard_Dataset(X)


vae_val_ds = _to_dataset(vae_val_ims, vae_val_meta, with_labels=True)
vae_val_loader = DataLoader(vae_val_ds, batch_size=VAE_params['batch_size'], shuffle=False)

inputmodule_paramsEnc = {'num_input_channels': 3}
configs_to_run = ['1']  # , '2', '3'

checkpoint_paths = [f'checkpoints/VAE_System{c}.pth' for c in configs_to_run]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Load Models into a List ---
loaded_models = []
for config_id, path in zip(configs_to_run, checkpoint_paths):
    if not os.path.exists(path):
        print(f"[Warning] checkpoint not found: {path} -- skipping config {config_id}")
        continue
    print(f"[Load] VAE Config {config_id} from {path}")

    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(config_id)

    # --- compute h_dim exactly as in VAE_training.py ---
    tmp_encoder = Encoder(inputmodule_paramsEnc, net_paramsEnc).to(device)
    tmp_encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(
            1,
            inputmodule_paramsEnc['num_input_channels'],
            VAE_params['img_size'][0],
            VAE_params['img_size'][1],
            device=device
        )
        h = tmp_encoder(dummy)
        h_dim = h.view(1, -1).size(1)

    net_paramsRep = {
        'h_dim': h_dim,
        'z_dim': 8,   # must match VAE_training.py
    }

    model = VAECNN(
        inputmodule_paramsEnc,
        net_paramsEnc,
        inputmodule_paramsDec,
        net_paramsDec,
        net_paramsRep
    )

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    loaded_models.append((config_id, model))

if len(loaded_models) == 0:
    raise RuntimeError("No VAE models loaded. Checkpoint files missing or paths incorrect.")

# We'll iterate once through the validation loader and for every model save reconstructions.
global_idx = 0  # tracks absolute index in the dataset
total_samples = len(vae_val_ds)

# For nicer progress bar:
pbar = tqdm(total=total_samples, desc="Saving reconstructions")

with torch.no_grad():
    for batch in vae_val_loader:
        # Standard_Dataset probably yields either x or (x, y) depending on implementation
        if isinstance(batch, (list, tuple)):
            x_batch = batch[0]
        else:
            x_batch = batch
        # ensure tensor on correct device and dtype
        x_batch = x_batch.to(dtype=torch.float32, device=device)

        batch_size = x_batch.shape[0]

        # For each model produce recon and save
        for config_id, model in loaded_models:
            # VAE forward → returns recon, mu, logvar
            recon_batch, _, _ = model(x_batch)  # (B, C, H, W)
            recon_batch = recon_batch.detach().cpu().clamp(0.0, 1.0)

            # Save each image individually mapping to meta
            for i in range(batch_size):
                idx = global_idx + i
                if idx >= len(vae_val_meta):
                    continue
                row = vae_val_meta.iloc[idx]
                patid = str(row['PatID'])
                filename = str(row['imfilename'])

                # create patient dir under RECON_DIR (optionally include config id subdir)
                out_dir = os.path.join(RECON_DIR, f"Config{config_id}", patid)
                os.makedirs(out_dir, exist_ok=True)

                # convert to HWC uint8
                img_tensor = recon_batch[i]  # (C, H, W)
                img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255.0).round().astype('uint8')  # (H,W,C)

                # Save with same filename
                out_path = os.path.join(out_dir, filename)
                try:
                    pil = Image.fromarray(img_np)
                    pil.save(out_path)
                except Exception as e:
                    print(f"[Save Error] idx {idx} file {out_path}: {e}")

        global_idx += batch_size
        pbar.update(batch_size)

pbar.close()
print("All reconstructions saved to:", RECON_DIR)