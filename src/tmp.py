import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import *
from Models.AEmodels import AutoEncoderCNN, AEWithLatent
from TripletLoss.triplet_loss import TripletLoss
from TripletLoss.datasets import TripletDataset
from Models.TripletModels import Embedder

from config import *

os.makedirs("/checkpoints/pepe", exist_ok=False)