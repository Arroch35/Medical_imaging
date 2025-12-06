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
from Models.datasets import Standard_Dataset, OnTheFlyImageDataset


from config import CROPPED_PATCHES_DIR, PATIENT_DIAGNOSIS_FILE
from utils import *

import wandb
import os

if __name__ == "__main__":
    # 0.1 AE PARAMETERS
    inputmodule_paramsEnc={}
    inputmodule_paramsEnc['num_input_channels']=3

    # 0.1 NETWORK TRAINING PARAMS
    AE_params = {
        'epochs': 30,
        'batch_size': 128,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'img_size': (256,256),   
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }


    #### 1. LOAD DATA: Implement 

    crossval_cropped_folders = get_all_subfolders(CROPPED_PATCHES_DIR)

    print("Loading images")

    training_metadata = GetImagePaths(list_folders=crossval_cropped_folders, n_images_per_folder=100,
                                    excelFile=PATIENT_DIAGNOSIS_FILE)
    print(training_metadata["path"][0])

    #### 3. lOAD PATCHES

    print("dataset")
    ae_training_dataset=OnTheFlyImageDataset(training_metadata)
    print("dataloader")
    ae_train_loader = DataLoader(ae_training_dataset, batch_size=AE_params['batch_size'], num_workers=4, shuffle=True)


    print(next(iter(ae_training_dataset)))
    print(next(iter(ae_train_loader)))

    # ### 4. AE TRAINING

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
            epoch_loss /= len(training_metadata)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[AE Config {Config}][Epoch {epoch+1}/{config_wandb.epochs}] loss={epoch_loss:.5f}")

            wandb.log({"train_loss": epoch_loss, "epoch": epoch+1})

        # Save model checkpoint
        Path('checkpoints').mkdir(exist_ok=True)
        checkpoint_path = f'checkpoints/fakeAE_Config{Config}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model for Config {Config} at {checkpoint_path}")

        # Free GPU memory after each config
        del model, optimizer, criterion
        torch.cuda.empty_cache()
        gc.collect()

        wandb.finish()


# ! Error: Ahora la loss es increiblemente grande!!!

# TODO: Quitar normalización de las imágenes y el clamp

# TODO: Con los últimos models aún me sale mal todo. Miara bien como lo ha hecho la jana, porque no es normal. Luego, cuando tenga algo bueno entrenado, usar el mejor para entrenar al enbeding CL, y despues al classifier