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
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import Standard_Dataset

from utils import *
from config import CROPPED_PATCHES_DIR, ANNOTATED_PATCHES_DIR, PATIENT_DIAGNOSIS_FILE, ANNOTATED_METADATA_FILE, RECON_DIR

from tqdm import tqdm
import wandb
import os


# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3

# 0.1 NETWORK TRAINING PARAMS
AE_params = {
    'epochs': 1,
    'batch_size': 256,
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

ae_val_ims, ae_val_meta = LoadAnnotated(
    annotated_folders, patient_excel=PATIENT_DIAGNOSIS_FILE,n_images_per_folder=None, 
    excelFile=ANNOTATED_METADATA_FILE, resize=AE_params['img_size'], verbose=True
)

# #### 3. lOAD PATCHES
ae_val_ds = to_dataset(ae_val_ims, ae_val_meta, with_labels=True)
ae_val_loader = DataLoader(ae_val_ds, batch_size=AE_params['batch_size'], shuffle=False)

inputmodule_paramsEnc = {'num_input_channels': 3} 
configs_to_run = ['1', '2', '3'] #

checkpoint_paths = [f'checkpoints/manual_remove/AE_Config{c}.pth' for c in configs_to_run]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Load Models into a List ---
loaded_models = []
for config_id, path in zip(configs_to_run, checkpoint_paths):
    if not os.path.exists(path):
        print(f"[Warning] checkpoint not found: {path} -- skipping config {config_id}")
        continue
    print(f"[Load] Config {config_id} from {path}")
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(config_id)
    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
    # map_location device ensures compatibility
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    loaded_models.append((config_id, model))

if len(loaded_models) == 0:
    raise RuntimeError("No models loaded. Checkpoint files missing.")

# We'll iterate once through the validation loader and for every model save reconstructions.
# Because DataLoader shuffle=False and dataset order matches ae_val_meta, we can map by global index.
global_idx = 0  # tracks absolute index in the dataset
total_samples = len(ae_val_ds)

# For nicer progress bar:
pbar = tqdm(total=total_samples, desc="Saving reconstructions")
errors=[]
with torch.no_grad():
    for batch in ae_val_loader:
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
            recon_batch = model(x_batch)  # (B, C, H, W)
            # Move to cpu and clamp
            recon_batch = recon_batch.detach().cpu().clamp(0.0, 1.0)

            # Save each image individually mapping to meta
            for i in range(batch_size):
                idx = global_idx + i
                if idx >= len(ae_val_meta):
                    # safety guard
                    continue
                row = ae_val_meta.iloc[idx]
                patid = str(row['PatID'])
                filename = str(row['imfilename'])

                # create patient dir under RECON_DIR (optionally include config id subdir)
                out_dir = os.path.join(RECON_DIR, "manual_remove", f"Config{config_id}", patid)
                os.makedirs(out_dir, exist_ok=True)

                # convert to HWC uint8
                img_tensor = recon_batch[i]  # (C, H, W)
                img_np = (img_tensor.numpy().transpose(1,2,0) * 255.0).round().astype('uint8')  # (H,W,C)

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