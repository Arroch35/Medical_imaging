import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
#import torchvision

import numpy as np
from random import shuffle
import random
from PIL import Image

    
# =============================================================================
# Standard dataset (Single Objective)
# =============================================================================
# X needs to have structure [NSamp,...] or be a list of NSamp entries
class Standard_Dataset(data.Dataset):
    def __init__(self, X, Y=None, transformation=None):
        super().__init__()
        self.X = X
        self.y = Y
        self.transformation = transformation
 
    def __len__(self):
        
        return len(self.X)

    def __getitem__(self, idx):
        
        if self.y is not None:
            return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(np.array(self.y[idx]))
        else:
            return torch.from_numpy(self.X[idx])

             
class OnTheFlyImageDataset(data.Dataset):
    def __init__(self, metadata_df, labels=None, resize=None, transform=None):
        """
        metadata_df: DataFrame with column 'path'
        labels: optional list/array aligned with metadata_df rows
        resize: tuple (H, W) or None
        transform: torchvision transforms
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.resize = resize
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.loc[idx, "path"]

        # Load image on the fly
        img = Image.open(img_path).convert("RGB")

        # Resize here if needed
        if self.resize is not None:
            img = img.resize(self.resize, Image.BILINEAR)

        # Apply torchvision transforms if provided
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2,0,1)

        if self.labels is not None:
            y = torch.tensor(self.labels[idx])
            return img, y
        
        return img
        





