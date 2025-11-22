# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:14:32 2024

Example of Main Steps for the Detection of HPilory using AutoEncoders for
the detection of anomalous pathological staining

Guides:
    0. Implement 2 functions for Loading Windows and metadata:
        0.1 LoadCropped to load a list of images from the Cropped folder
            inputs: list of folders containing the images, number of images to load for each folder,
                    ExcelFile with metadata
            out: Ims: list of images
                 metadata: list/array of information for each image in Ims
                           (PatID, imfilename)
        0.1 LoadAnnotated to load a list of images from the Annotated folder
            inputs: list of folders containing the images, number of images to load for each folder,
                    ExcelFile with metadata
            out: Ims: list of images
                 metadata: list/array of information for each image in Ims
                           (PatID, imfilename,presenceHelico)
                           
    1. Split Code into train and test steps 
    2. Save trainned models and any intermediate result input of the next step
    
@authors: debora gil, pau cano
email: debora@cvc.uab.es, pcano@cvc.uab.es
Reference: https://arxiv.org/abs/2309.16053 

"""
# IO Libraries
import sys
import os
import pickle
from pathlib import Path

# Standard Libraries
import numpy as np
import pandas as pd
import glob
from PIL import Image
from tqdm import tqdm

# Torch Libraries
from torch.utils.data import DataLoader
import gc
import torch
import torch.nn as nn
import torch.optim as optim


# Own Functions
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import Standard_Dataset


from config import CROPPED_PATCHES_DIR, ANNOTATED_PATCHES_DIR, PATIENT_DIAGNOSIS_FILE, ANNOTATED_METADATA_FILE

import wandb
import os

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

def _read_metadata_excel(excel_path):
    """
    Read an excel metadata file and return a DataFrame.
    If excel_path is None or doesn't exist -> returns empty DataFrame.
    """
    if excel_path is None:
        return pd.DataFrame()
    if not os.path.exists(excel_path):
        print(f"[Warning] Metadata file not found: {excel_path}")
        return pd.DataFrame()
    try:
        df = pd.read_excel(excel_path)
    except Exception:
        # Try CSV fallback
        try:
            df = pd.read_csv(excel_path)
        except Exception:
            print(f"[Warning] Could not read metadata file: {excel_path}")
            return pd.DataFrame()
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df


def _window_id_from_filename(fname):
    """
    Normalize a filename to a Window_ID style used in excel:
    strips extension and returns base name.
    """
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    return name

def LoadCropped(list_folders, n_images_per_folder=None, excelFile=None, resize=None, verbose=False):
    """
    Load cropped patches (non-annotated), optionally resizing images.
    If a PatientDiagnosis.csv file (with columns CODI, DENSITAT) is provided,
    only patches from healthy patients (DENSITAT == 'NEGATIVA') are loaded.

    Parameters
    ----------
    list_folders : list of str
        Paths containing .png patches (e.g. PatID_Section# folders)
    n_images_per_folder : int or None
        Limit number of patches per folder (first N if provided)
    excelFile : str or None
        Path to PatientDiagnosis.csv (columns: CODI, DENSITAT)
    resize : tuple(int, int) or None
        Target image size (H, W). If None, keep original 256x256.
    verbose : bool
        Print progress messages if True.

    Returns
    -------
    Ims : np.ndarray
        Array of images with shape (N, H, W, 3), dtype=uint8
    metadata : pd.DataFrame
        DataFrame with columns ['PatID', 'imfilename']
    """
    
    healthy_pats = None
    all_patids = set()
    skipped_patids = set()

    if excelFile is not None:
        if not os.path.exists(excelFile):
            raise FileNotFoundError(f"[LoadCropped] File not found: {excelFile}")

        try:
            df_diag = pd.read_csv(excelFile)
        except Exception:
            df_diag = pd.read_excel(excelFile)

        # Normalize column names
        df_diag.columns = [c.strip().upper() for c in df_diag.columns]
        if not {"CODI", "DENSITAT"}.issubset(df_diag.columns):
            raise ValueError("Diagnosis file must contain columns: 'CODI' and 'DENSITAT'")

        # Select healthy patients: DENSITAT == 'NEGATIVA'
        healthy_mask = df_diag["DENSITAT"].astype(str).str.upper().str.strip() == "NEGATIVA"
        healthy_pats = set(df_diag.loc[healthy_mask, "CODI"].astype(str))
        if verbose:
            print(f"[LoadCropped] Found {len(healthy_pats)} healthy patients in {excelFile}")


    records, images = [], []

    for folder in list_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        patid = folder_name.split("_")[0]  
        all_patids.add(patid)

        # Skip if not a healthy patient (if CSV provided)
        if healthy_pats is not None and patid not in healthy_pats:
            skipped_patids.add(patid)
            if verbose:
                print(f"[LoadCropped] Skipping non-healthy patient: {patid}")
            continue

        if not os.path.isdir(folder):
            if verbose:
                print(f"[LoadCropped] Folder not found, skipping: {folder}")
            continue

        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if n_images_per_folder is not None:
            files = files[:n_images_per_folder]

        for fpath in files:
            try:
                im = Image.open(fpath).convert("RGB")
                if resize is not None:
                    im = im.resize(resize, Image.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)
            except Exception as e:
                if verbose:
                    print(f"[LoadCropped] Failed to read {fpath}: {e}")
                continue

            filename = os.path.basename(fpath)
            record = {"PatID": patid, "imfilename": filename}
            records.append(record)
            images.append(arr)


    # Calculate percentage of non-healthy patients
    total_patients = len(all_patids)
    non_healthy_count = len(skipped_patids)
    if total_patients > 0:
        perc_non_healthy = 100 * non_healthy_count / total_patients
    else:
        perc_non_healthy = 0.0

    if verbose:
        print(f"[LoadCropped] Total patients found: {total_patients}")
        print(f"[LoadCropped] Non-healthy patients skipped: {non_healthy_count} ({perc_non_healthy:.2f}%)")
        print(f"[LoadCropped] Loaded {len(images)} images from {len(records)} healthy patients.")


    if len(images) == 0:
        H, W = resize if resize is not None else (256, 256)
        Ims = np.zeros((0, H, W, 3), dtype=np.uint8)
    else:
        Ims = np.stack(images, axis=0).astype(np.uint8)

    metadata = pd.DataFrame.from_records(records)
    if verbose:
        print(f"[LoadCropped] Loaded {len(images)} images from {len(metadata['PatID'].unique())} healthy patients.")
    return Ims, metadata


def LoadAnnotated(list_folders, n_images_per_folder=None, excelFile=None, resize=None, verbose=False):
    """
    Load annotated patches (presence of H. pylori known), optionally resizing images.

    Parameters
    ----------
    list_folders : list
        Paths containing annotated .png patches
    n_images_per_folder : int or None
        Limit number of patches per folder (first N if provided)
    excelFile : str or None
        Metadata Excel with 'Pat_ID', 'Window_ID', 'Presence' columns
    resize : tuple(int,int) or None
        Target image size (H, W). If None, keep original 256x256.
    verbose : bool
        Print progress messages if True

    Returns
    -------
    Ims : np.ndarray
        Images as (N, H, W, 3) array, dtype=uint8
    metadata : pd.DataFrame
        DataFrame with columns ['PatID', 'imfilename', 'Presence']
    """
    df = _read_metadata_excel(excelFile)
    records = []
    images = []

    for folder in list_folders:
        if not os.path.isdir(folder):
            if verbose:
                print(f"[LoadAnnotated] folder not found, skipping: {folder}")
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if n_images_per_folder is not None:
            files = files[:n_images_per_folder]

        for fpath in files:
            try:
                im = Image.open(fpath).convert("RGB")
                if resize is not None:
                    im = im.resize(resize, Image.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)
            except Exception as e:
                if verbose:
                    print(f"[LoadAnnotated] failed to read {fpath}: {e}")
                continue

            filename = os.path.basename(fpath)
            window_id = _window_id_from_filename(filename)
            folder_name = os.path.basename(os.path.normpath(folder))
            if "_" in folder_name:
                patid, section = folder_name.split("_", 1)
            else:
                patid = folder_name
                section = ""

            presence_val = None
            if not df.empty:
                if 'Aug' in window_id:
                    parts = window_id.split('_')
                    window_id_clean = f"{int(parts[0])}_{parts[1]}"
                else:
                    try:
                        window_id_clean = str(int(window_id))
                    except ValueError:
                        window_id_clean = window_id

                m = df[df['Window_ID'].astype(str).str.strip() == window_id_clean]
                if m.empty and 'Pat_ID' in df.columns:
                    m = df[(df['Window_ID'].astype(str).str.strip() == window_id_clean)
                           & (df['Pat_ID'].astype(str).str.strip() == patid)]
                if m.empty:
                    m = df[df['Window_ID'].astype(str).str.strip() == filename]
                if not m.empty and 'Presence' in m.columns:
                    presence_val = m.iloc[0]['Presence']

                if presence_val == 0 or presence_val == None:
                    continue

            record = {'PatID': patid, 'imfilename': filename, 'Presence': presence_val}
            images.append(arr)
            records.append(record)

    if len(images) == 0:
        H, W = resize if resize is not None else (256, 256)
        Ims = np.zeros((0, H, W, 3), dtype=np.uint8)
    else:
        Ims = np.stack(images, axis=0).astype(np.uint8)

    metadata = pd.DataFrame.from_records(records)
    return Ims, metadata

def get_all_subfolders(root_dir):
    """
    Return a sorted list of all subfolders in root_dir (recursively).
    Each folder typically corresponds to one patient/section.
    """
    subfolders = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            subfolders.append(os.path.join(root, d))
    subfolders = sorted(subfolders)
    return subfolders




# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3

# 0.1 NETWORK TRAINING PARAMS
AE_params = {
    'epochs': 30,
    'batch_size': 2048,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'img_size': (256,256),   
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


#### 1. LOAD DATA: Implement 
# 1.1 Patient Diagnosis
df_diag = pd.read_csv(PATIENT_DIAGNOSIS_FILE) if os.path.isfile(PATIENT_DIAGNOSIS_FILE) else None


# 1.2 Patches Data

crossval_cropped_folders = get_all_subfolders(CROPPED_PATCHES_DIR)

print("Loading images")

ae_train_ims, ae_train_meta = LoadCropped(
    crossval_cropped_folders, n_images_per_folder=500, 
    excelFile=PATIENT_DIAGNOSIS_FILE, resize=AE_params['img_size'], verbose=True
)


#### 3. lOAD PATCHES
def _to_dataset(ims, meta, with_labels=False):
    X = np.stack([im.transpose(2,0,1) for im in ims], axis=0) / 255.0 
    if with_labels:
        y = meta['Presence'].to_numpy(dtype=np.int64) 
        return Standard_Dataset(X, y)
    else:
        return Standard_Dataset(X)

print("dataset")
ae_train_ds = _to_dataset(ae_train_ims, ae_train_meta, with_labels=False)
print("dataloader")
ae_train_loader = DataLoader(ae_train_ds, batch_size=AE_params['batch_size'], num_workers=4, shuffle=True)

### 4. AE TRAINING

configs_to_run = ['1', '2', '3']
print(AE_params['device'])
for Config in configs_to_run:
    print(f"\n===== Training AE with CONFIG {Config} =====")

    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                           inputmodule_paramsDec, net_paramsDec)
    model.to(AE_params['device'])

    # Initialize W&B run for this configuration
    wandb.init(
        project="autoencoder_hp",
        name=f"AE_Config{Config}",
        dir=f"wandb_runs/AE_Config{Config}", 
        mode="offline",
        config={
            "epochs": AE_params['epochs'],
            "batch_size": AE_params['batch_size'],
            "lr": AE_params['lr'],
            "weight_decay": AE_params['weight_decay'],
            "img_size": AE_params['img_size'],
            "model": "AutoEncoderCNN",
            "optimizer": "Adam",
            "config": Config
        }
    )
    config_wandb = wandb.config

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config_wandb.lr, weight_decay=config_wandb.weight_decay)
    criterion = nn.MSELoss()
    

    # Training loop
    model.train()
    for epoch in range(config_wandb.epochs):
        epoch_loss = 0.0
        for batch in tqdm(ae_train_loader, desc=f"Epoch {epoch+1}/{config_wandb.epochs}"):
            x = batch.to(torch.float32).to(AE_params['device'])
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(ae_train_ds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[AE Config {Config}][Epoch {epoch+1}/{config_wandb.epochs}] loss={epoch_loss:.5f}")

        wandb.log({"train_loss": epoch_loss, "epoch": epoch+1})

    # Save model checkpoint
    Path('checkpoints').mkdir(exist_ok=True)
    checkpoint_path = f'checkpoints/AE_Config{Config}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model for Config {Config} at {checkpoint_path}")

    # Free GPU memory after each config
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

    wandb.finish()