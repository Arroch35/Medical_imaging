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


def LoadAnnotated(list_folders, patient_excel, n_images_per_folder=None, excelFile=None, resize=None, verbose=False):
    """
    Load annotated patches for patients with **non-negative diagnosis**, and compute the percentage.

    Parameters
    ----------
    list_folders : list
        Paths containing annotated .png patches
    patient_excel : str
        Excel/CSV file with patient diagnoses (columns: 'CODI', 'DENSITAT')
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
    # Load patient diagnosis file
    if not os.path.exists(patient_excel):
        raise FileNotFoundError(f"[LoadAnnotatedPositivePatients] Patient file not found: {patient_excel}")
    
    try:
        df_patients = pd.read_csv(patient_excel)
    except Exception:
        df_patients = pd.read_excel(patient_excel)

    df_patients.columns = [c.strip().upper() for c in df_patients.columns]
    if not {"CODI", "DENSITAT"}.issubset(df_patients.columns):
        raise ValueError("Patient file must contain columns: 'CODI' and 'DENSITAT'")

    # Keep only non-negative patients
    non_negative_pats = set(df_patients.loc[df_patients['DENSITAT'].str.upper().str.strip() != "NEGATIVA", "CODI"].astype(str))

    # Track all patients in the folders
    all_patients = set()
    for folder in list_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        patid = folder_name.split("_")[0] if "_" in folder_name else folder_name
        all_patients.add(patid)

    # Compute percentage
    total_patients = len(all_patients)
    non_negative_count = len([p for p in all_patients if p in non_negative_pats])
    perc_non_negative = 100 * non_negative_count / total_patients if total_patients > 0 else 0.0
    if verbose:
        print(f"[LoadAnnotated] Total patients found: {total_patients}")
        print(f"[LoadAnnotated] Non-negative patients: {non_negative_count} ({perc_non_negative:.2f}%)")

    # Load annotation metadata
    df = _read_metadata_excel(excelFile)
    records = []
    images = []

    for folder in list_folders:
        if not os.path.isdir(folder):
            if verbose:
                print(f"[LoadAnnotated] folder not found, skipping: {folder}")
            continue

        folder_name = os.path.basename(os.path.normpath(folder))
        patid = folder_name.split("_")[0] if "_" in folder_name else folder_name

        # # Skip if patient is negative
        # if patid not in non_negative_pats:
        #     continue

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

            # Get presence if excelFile is provided
            presence_val = None
            if df is not None and not df.empty:
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
                    m = df[(df['Window_ID'].astype(str).str.strip() == window_id_clean) &
                           (df['Pat_ID'].astype(str).str.strip() == patid)]
                if not m.empty and 'Presence' in m.columns:
                    presence_val = m.iloc[0]['Presence']

                if presence_val not in [-1, 1]:
                    continue
                if presence_val == -1: presence_val = 0 #ROC curve expects 0/1 values

            record = {'PatID': patid, 'imfilename': filename, 'Presence': presence_val}
            images.append(arr)
            records.append(record)

    if len(images) == 0:
        H, W = resize if resize is not None else (256, 256)
        Ims = np.zeros((0, H, W, 3), dtype=np.uint8)
    else:
        Ims = np.stack(images, axis=0).astype(np.uint8)

    metadata = pd.DataFrame.from_records(records)
    if verbose:
        print(f"[LoadAnnotated] Loaded {len(images)} images from {len(metadata['PatID'].unique())} non-negative patients.")
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


#### 3. lOAD PATCHES
def to_dataset(ims, meta, with_labels=False):
    X = np.stack([im.transpose(2,0,1) for im in ims], axis=0) / 255.0 
    if with_labels:
        y = meta['Presence'].to_numpy(dtype=np.int64) 
        return Standard_Dataset(X, y)
    else:
        return Standard_Dataset(X)

ae_val_ds = to_dataset(ae_val_ims, ae_val_meta, with_labels=True)
ae_val_loader = DataLoader(ae_val_ds, batch_size=AE_params['batch_size'], shuffle=False)

# Evaluation:

inputmodule_paramsEnc = {'num_input_channels': 3} 
configs_to_run = ['1', '2', '3']

checkpoint_paths = [f'checkpoints/AE_Config{c}.pth' for c in configs_to_run]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Load Models into a List ---
loaded_models = []
model_configs = [] # To keep track of which config belongs to which model
ae_losses_dict = {}

y_true_list = []
for _, y in ae_val_loader:
    y_true_list.extend(y.numpy())
y_true = np.array(y_true_list)

for config_id, path in zip(configs_to_run, checkpoint_paths):
    print(f"Loading model for Config {config_id} from {path}...")
    
    # 1. Instantiate the correct architecture for this config
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(config_id)
    
    model = AutoEncoderCNN(
        inputmodule_paramsEnc, net_paramsEnc,
        inputmodule_paramsDec, net_paramsDec
    )

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode (important for inference/evaluation)
    
    loaded_models.append(model)
    model_configs.append(config_id)
    print(f"Successfully loaded Config {config_id}.")


    all_losses = []

    with torch.no_grad():
        for batch in ae_val_loader:  # validation DataLoader
            x = batch[0].to(torch.float32).to(AE_params['device'])  # assuming batch = (X,) or (X, y)
            recon = model(x)
            # Compute per-sample MSE
            batch_losses = ((x - recon)**2).mean(dim=[1,2,3])  # MSE per image
            all_losses.extend(batch_losses.cpu().numpy())

    all_losses = np.array(all_losses)
    ae_losses_dict[path] = all_losses

    # y_true = []
    # for _, y in ae_val_loader:
    #     y_true.extend(y.numpy())
    # y_true = np.array(y_true)

    print(f"losses: {all_losses[0]}, y_true: {y_true[0]}")
    print(set(y_true))
    # ROC curve

    # order scores
    # order = np.argsort(all_losses)
    # all_losses = all_losses[order]
    # y_true = y_true[order]

    fpr, tpr, thresholds = roc_curve(y_true, all_losses)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')  # diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for AE Reconstruction')
    plt.legend()
    plt.show()


    #Decide threshold
    # Option 1: You can pick threshold at maximum Youden’s J statistic
    J = tpr - fpr
    idx = np.argmax(J)
    best_threshold = thresholds[idx]
    print("Best threshold:", best_threshold)

    # Apply threshold
    y_pred = (all_losses > best_threshold).astype(int)


# Compare multiple AEs
plt.figure(figsize=(6,6))
for ae_name, losses in ae_losses_dict.items():
    fpr, tpr, _ = roc_curve(y_true, losses)
    plt.plot(fpr, tpr, label=f'{ae_name} (AUC={auc(fpr, tpr):.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()


#TODO:
#! Algo esta mal porque las ROC curves me salen difernte en cada repeticion, cosa que no tendria que pasar 
# ? Ordenar el valor de los losses para que la ROC curve este bien
# Hacer un k-fold stratified con shuffle = false para la validacion usaar sk-learn
# Hacer box plots para la k-fold
# Hacer las ROC curves conRED channel

#? Config 3 seems to be the best one. train with different hyperparams to see if it improves
#? Puede que el image size esté afectando. probar diferentes metricas y bajar el numero de imagenes por paciente