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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# your provided dataset wrapper
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle as sk_shuffle
import json


## Own Functions
from Models.VAEmodels import VAECNN, Encoder
from Models.datasets import Standard_Dataset
from utils import LoadAnnotated, get_all_subfolders
from config2 import CROPPED_PATCHES_DIR, ANNOTATED_PATCHES_DIR, PATIENT_DIAGNOSIS_FILE, ANNOTATED_METADATA_FILE


def VAEConfigs(Config):
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


# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3

# 0.1 NETWORK TRAINING PARAMS
VAE_params = {
    'epochs': 15,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'img_size': (256,256),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# 0.2 FOLDERS

#### 1. LOAD DATA: Implement

# 1.1 Patient Diagnosis
df_diag = pd.read_csv(PATIENT_DIAGNOSIS_FILE) if os.path.isfile(PATIENT_DIAGNOSIS_FILE) else None

# 1.2 Patches Data
annotated_folders = get_all_subfolders(ANNOTATED_PATCHES_DIR)

vae_val_ims, vae_val_meta = LoadAnnotated(
    annotated_folders, patient_excel=PATIENT_DIAGNOSIS_FILE,n_images_per_folder=None,
    excelFile=ANNOTATED_METADATA_FILE, resize=VAE_params['img_size'], verbose=True
)

# check the dtype of vae_val_meta
print(f'dtype of vae_val_meta: {type(vae_val_meta)}')
print(f"Metadata 'Presence' column dtype: {vae_val_meta['Presence'].dtype}")

#### 3. lOAD PATCHES
def _to_dataset(ims, meta, with_labels=False):
    X = np.stack([im.transpose(2,0,1) for im in ims], axis=0) / 255.0
    if with_labels:
        y = meta['Presence'].to_numpy(dtype=np.int64)
        return Standard_Dataset(X, y)
    else:
        return Standard_Dataset(X)

vae_val_ds = _to_dataset(vae_val_ims, vae_val_meta, with_labels=True)
vae_val_loader = DataLoader(vae_val_ds, batch_size=VAE_params['batch_size'], shuffle=False)

inputmodule_paramsEnc = {'num_input_channels': 3}
configs_to_run = ['1', '2', '3']

checkpoint_paths = [f'checkpoints/VAE_System{c}.pth' for c in configs_to_run]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Load Models into a List ---
loaded_models = []
model_configs = [] # To keep track of which config belongs to which model
vae_losses_dict = {}

y_true = []
for _, y in vae_val_loader:
    y_true.extend(y.numpy())
y_true = np.array(y_true)

for config_id, path in zip(configs_to_run, checkpoint_paths):
    print(f"Loading model for Config {config_id} from {path}...")
    
    # determine h_dim by passing a dummy input through the encoder
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = VAEConfigs(config_id)

    tmp_encoder = Encoder(inputmodule_paramsEnc, net_paramsEnc)
    tmp_encoder.eval()

    # pass a dummy input to determine h_dim
    with torch.no_grad():
        dummy = torch.zeros(
            1,
            inputmodule_paramsEnc['num_input_channels'],
            VAE_params['img_size'][0],
            VAE_params['img_size'][1],
        )
        h = tmp_encoder(dummy)  # (1, C', H', W')
        h_dim = h.view(1, -1).size(1)  # flatten → size h_dim

    # define the parameters for the bottleneck representation
    net_paramsRep = {
        'h_dim': h_dim,
        'z_dim': 8,
    }

    model = VAECNN(inputmodule_paramsEnc, net_paramsEnc,
                   inputmodule_paramsDec, net_paramsDec,
                   net_paramsRep)

    # load state dict
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode (important for inference/evaluation)

    loaded_models.append(model)
    model_configs.append(config_id)
    print(f"Successfully loaded Config {config_id}.")

    # list to store all reconstruction losses
    all_losses = []

    # evaluate on validation set
    with torch.no_grad():
        for batch in vae_val_loader:
            # batch = (X, y)
            x = batch[0].to(torch.float32).to(VAE_params['device'])
            x_recon, mu, logvar = model(x)
            # per-sample MSE
            batch_losses = ((x - x_recon) ** 2).mean(dim=[1, 2, 3])
            all_losses.extend(batch_losses.cpu().numpy())

    all_losses = np.array(all_losses)
    vae_losses_dict[path] = all_losses

    print(f"losses: {all_losses[0]}, y_true: {y_true[0]}")
    print(set(y_true))

    # ROC curve
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
    J = tpr - fpr
    idx = np.argmax(J)
    best_threshold = thresholds[idx]
    print("Best threshold:", best_threshold)

    # Apply threshold
    y_pred = (all_losses > best_threshold).astype(int)

# Compare multiple AEs and save the plot
plt.figure(figsize=(6,6))
for ae_name, losses in vae_losses_dict.items():
    fpr, tpr, _ = roc_curve(y_true, losses)
    plt.plot(fpr, tpr, label=f'{ae_name} (AUC={auc(fpr, tpr):.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()


#TODO: 
# Hacer aquí un training simple de los tres models con pocas imagenes para probar el evaluation
# Acabar el evaluation con lo de la ROC curve y threshols y tal (todo esta en el chatgpt) 
# ? Tengo que evitar tambien que se cojan imagenes con presence = 0, porque esto es que no estan anotados. Mirar primero si hay presencia 0.0