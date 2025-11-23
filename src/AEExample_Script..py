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
from Models.VAEmodels import VAECNN, Encoder
from Models.datasets import Standard_Dataset

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

        # Skip if not a healthy patient (if CSV provided)
        if healthy_pats is not None and patid not in healthy_pats:
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


crossval_cropped_folders = get_all_subfolders(CROPPED_PATCHES_DIR)
annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)


print(f"Found {len(crossval_cropped_folders)} cropped folders and {len(annotated_folders)} annotated folders.")


# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3

# 0.1 NETWORK TRAINING PARAMS
VAE_params = {
    'epochs': 15,
    'batch_size': 64,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'img_size': (128,128),   
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# 0.2 FOLDERS

#### 1. LOAD DATA: Implement

# 1.1 Patient Diagnosis
df_diag = pd.read_csv(PATIENT_DIAGNOSIS_FILE) if os.path.isfile(PATIENT_DIAGNOSIS_FILE) else None


# 1.2 Patches Data
ae_train_ims, ae_train_meta = LoadCropped(
    crossval_cropped_folders, n_images_per_folder=5, excelFile=PATIENT_DIAGNOSIS_FILE,
    resize=VAE_params['img_size']
)
print("Cropped loaded:", ae_train_ims.shape, ae_train_meta.shape)
print(ae_train_meta.head())


# Annotated para aprender umbral de error (ROC)
ann_ims, ann_meta = LoadAnnotated(
    annotated_folders, n_images_per_folder=5, excelFile=ANNOTATED_METADATA_FILE,
    resize=VAE_params['img_size']
)
print("Annotated loaded:", ann_ims.shape, ann_meta.shape)
print(ann_meta.head())


print(f"Found {len(crossval_cropped_folders)} cropped folders and {len(annotated_folders)} annotated folders.")


#### 2. DATA SPLITING INTO INDEPENDENT SETS

# 2.0 Annotated set for FRed optimal threshold
# later 
# 2.1 AE trainnig set

# 2.1 Diagosis crossvalidation set

#### 3. lOAD PATCHES

#this function takes images and metadata and returns a Standard_Dataset object 
def _to_dataset(ims, meta, with_labels=False):
    X = np.stack([im.transpose(2,0,1) for im in ims], axis=0) / 255.0  #? Puede que esté aquí el problema de las dimensiones?
    if with_labels:
        y = np.array([m['Presence'] for m in meta], dtype=np.int64)
        return Standard_Dataset(X, y)
    else:
        return Standard_Dataset(X)

ae_train_ds = _to_dataset(ae_train_ims, ae_train_meta, with_labels=False)
ae_train_loader = DataLoader(ae_train_ds, batch_size=VAE_params['batch_size'],
                             shuffle=True)

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
Config='2'
net_paramsEnc,net_paramsDec,inputmodule_paramsDec=AEConfigs(Config)

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
        # with torch.no_grad():
        #     print(f"mu mean={mu.mean().item():.4f}, mu std={mu.std().item():.4f}, "
        #         f"logvar mean={logvar.mean().item():.4f}, logvar std={logvar.std().item():.4f}")


    epoch_loss /= len(ae_train_ds)
    epoch_recon /= len(ae_train_ds)
    epoch_kl /= len(ae_train_ds)

    print(f"[VAE][Epoch {epoch+1}/{VAE_params['epochs']}] loss={epoch_loss:.5f} | recon={epoch_recon:.5f} | kld={epoch_kl:.5f}")


Path('checkpoints').mkdir(exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/VAE_System2.pth') # save model 

# Free GPU Memory After Training
gc.collect()
torch.cuda.empty_cache()