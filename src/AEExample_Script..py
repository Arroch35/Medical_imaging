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

# Torch Libraries
#from torch.utils.data import DataLoader
#import gc
#import torch


## Own Functions
#from Models.AEmodels import AutoEncoderCNN

from config import CROPPED_PATCHES_DIR, ANNOTATED_PATCHES_DIR, PATIENT_DIAGNOSIS_FILE, ANNOTATED_METADATA_FILE

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


def LoadCropped(list_folders, n_images_per_folder=None, excelFile=None, verbose=False):
    """
    Load cropped patches (non-annotated), returning images and minimal metadata.

    Inputs:
      - list_folders: list of folder paths where each folder contains .png patches
                      (folder typically named PatID_Section#)
      - n_images_per_folder: int or None, if int limits images per folder (first N sorted)
      - excelFile: path to a metadata excel/csv (optional). If provided, will try to attach
                   matching rows from the excel based on Window_ID or filename.
    Outputs:
      - Ims: np.array of shape (N, H, W, C) dtype=float32, values [0..255]
      - metadata: pandas.DataFrame with columns at least:
          ['PatID', 'Section', 'filename', 'Window_ID'] plus extra columns from excel if matched
    """
    df_meta = _read_metadata_excel(excelFile)
    records = []
    images = []

    for folder in list_folders:
        if not os.path.isdir(folder):
            if verbose:
                print(f"[LoadCropped] folder not found, skipping: {folder}")
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if n_images_per_folder is not None:
            files = files[:n_images_per_folder]

        for fpath in files:
            try:
                im = Image.open(fpath).convert("RGB")
                arr = np.asarray(im, dtype=np.uint8)
            except Exception as e:
                if verbose:
                    print(f"[LoadCropped] failed to read {fpath}: {e}")
                continue

            filename = os.path.basename(fpath)
            window_id = _window_id_from_filename(filename)
            # parse PatID and Section from folder name if possible: PatID_Section
            folder_name = os.path.basename(os.path.normpath(folder))
            if "_" in folder_name:
                patid, section = folder_name.split("_", 1)
            else:
                patid = folder_name
                section = ""

            record = {'PatID': patid, 'imfilename': filename}
            
            images.append(arr)
            records.append(record)

    if len(images) == 0:
        Ims = np.zeros((0, 256, 256, 3), dtype=np.uint8)
    else:
        Ims = np.stack(images, axis=0).astype(np.uint8)

    metadata = pd.DataFrame.from_records(records)
    return Ims, metadata


def LoadAnnotated(list_folders, n_images_per_folder=None, excelFile=None, verbose=False):
    """
    Load annotated patches (where presence of H. pylori has been marked).

    Inputs:
      - list_folders: list of folder paths with annotated .png patches
      - n_images_per_folder: optional int limit
      - excelFile: path to annotation excel that contains columns including
                   ['Pat_ID'/'Pat_ID', 'Window_ID', 'Presence'] (presence values like 1, -1, 0)
    Outputs:
      - Ims: np.array of images (N,H,W,C)
      - metadata: pandas.DataFrame with columns:
          ['PatID', 'Section', 'filename', 'Window_ID', 'Presence'] and extra excel columns if present
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
                arr = np.asarray(im, dtype=np.uint8)
            except Exception as e:
                if verbose:
                    print(f"[LoadAnnotated] failed to read {fpath}: {e}")
                continue

            filename = os.path.basename(fpath)
            window_id = _window_id_from_filename(filename)
            # parse folder name
            folder_name = os.path.basename(os.path.normpath(folder))
            if "_" in folder_name:
                patid, section = folder_name.split("_", 1)
            else:
                patid = folder_name
                section = ""

            # find annotation in excel if possible
            presence_val = None
            extra_info = {}
            if not df.empty:
                if 'Aug' in window_id:
                    window_id_without_0s=window_id.split('_')
                    window_id_without_0s =  str(int(window_id_without_0s[0]))+'_'+window_id_without_0s[1]
                else:
                    window_id_without_0s = str(int(window_id))
                # Try matches using several keys
                # match by Window_ID
                m = df[df['Window_ID'].astype(str).str.strip() == window_id_without_0s]
                # try Pat_ID + Window_ID if available
                if m.empty and 'Pat_ID' in df.columns:
                    m = df[(df['Window_ID'].astype(str).str.strip() == window_id_without_0s) & (df['Pat_ID'].astype(str).str.strip() == patid)]
                # try matching by filename with extension
                if m.empty:
                    m = df[df['Window_ID'].astype(str).str.strip() == filename]
                if not m.empty:
                    # take the first
                    row = m.iloc[0]
                    if 'Presence' in row.index:
                        presence_val = row['Presence']
                        

            record = {
                'PatID': patid,
                'imfilename': filename,
                'Presence': presence_val
            }
            # attach extras
            record.update(extra_info)

            images.append(arr)
            records.append(record)

    if len(images) == 0:
        Ims = np.zeros((0, 256, 256, 3), dtype=np.uint8)
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


I_annot, meta_annot = LoadAnnotated(annotated_folders, n_images_per_folder=5, excelFile=ANNOTATED_METADATA_FILE, verbose=True)
print("Annotated loaded:", I_annot.shape, meta_annot.shape)
print(meta_annot.head())

# Load some cropped (non-annotated) patches for AE training (healthy)
I_cropped, meta_cropped = LoadCropped(crossval_cropped_folders, n_images_per_folder=5, excelFile=None, verbose=True)
print("Cropped loaded:", I_cropped.shape, meta_cropped.shape)
print(meta_cropped.head())

#### 1. LOAD DATA: Implement 
# 1.1 Patient Diagnosis

# 1.2 Patches Data


#### 2. DATA SPLITING INTO INDEPENDENT SETS

# 2.0 Annotated set for FRed optimal threshold

# 2.1 AE trainnig set

# 2.1 Diagosis crossvalidation set

#### 3. lOAD PATCHES

### 4. AE TRAINING

# EXPERIMENTAL DESIGN:
# TRAIN ON AE PATIENTS AN AUTOENCODER, USE THE ANNOTATED PATIENTS TO SET THE
# THRESHOLD ON FRED, VALIDATE FRED FOR DIAGNOSIS ON A 10 FOLD SCHEME OF REMAINING
# CASES.

# 4.1 Data Split


# ###### CONFIG1
# Config='1'
# net_paramsEnc,net_paramsDec,inputmodule_paramsDec=AEConfigs(Config)
# model=AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
#                      inputmodule_paramsDec, net_paramsDec)
# # 4.2 Model Training

# # Free GPU Memory After Training
# gc.collect()
# torch.cuda.empty_cache()
# #### 5. AE RED METRICS THRESHOLD LEARNING

# ## 5.1 AE Model Evaluation

# # Free GPU Memory After Evaluation
# gc.collect()
# torch.cuda.empty_cache()

# ## 5.2 RedMetrics Threshold 

# ### 6. DIAGNOSIS CROSSVALIDATION
# ### 6.1 Load Patches 4 CrossValidation of Diagnosis

# ### 6.2 Diagnostic Power

