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
import cv2


# Own Functions
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import Standard_Dataset


from config import CROPPED_PATCHES_DIR, ANNOTATED_PATCHES_DIR, PATIENT_DIAGNOSIS_FILE, ANNOTATED_METADATA_FILE



df_anno=pd.read_excel(ANNOTATED_METADATA_FILE)  

# Assuming you are in the Annotated folder

os.chdir(ANNOTATED_PATCHES_DIR)
 
AnnoSet2=[]
AnnoSetLab2=[] 

for k in np.arange(len(df_anno)):  
    folder=df_anno.Pat_ID.values[k]+'_'+str(df_anno.Section_ID.values[k])
    if isinstance(df_anno.Window_ID.values[k],str):
       imfile= str(df_anno.Window_ID.values[k].split('_')[0]).zfill(5)+'_'+df_anno.Window_ID.values[k].split('_')[1]
       img = cv2.imread(os.path.join(folder,imfile+'.png'))
       label=1
    else:
       imfile= str(df_anno.Window_ID.values[k]).zfill(5)
       img = cv2.imread(os.path.join(folder,imfile+'.png'))
       label= df_anno.Presence.values[k]
    if img is not None:
        AnnoSet2.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        AnnoSetLab2.append(label)