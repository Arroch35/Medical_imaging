# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:19:56 2023

@author: debora
"""
import torch
from PreTrainedModels import *
import numpy as np
from ismember import ismember

from skimage import measure

from datasets import create_dataloader
from test_models import eval_model

def FeatExtractor(ims,Model,batch_size):
  

  # Instantation of Feature Extractor 
  
  if (Model == 'Vgg16' or Model == 'Vgg19'):   net = FeatureExtractorVGG(Model, cut_index=3)
  if (Model == 'DenseNet'):   net = FeatureExtractorDENSENET169(Model)
  if (Model == 'ResNet'):   net = FeatureExtractorResnet152(Model)
  if (Model == 'EfficientNet'):   net = FeatureExtractorEFFICIENTNETB7(Model)
  
  net.to('cuda')
  net.eval()
  
  fe=[]
  
  test_dataloader = create_dataloader(ims, np.ones(ims.shape[0]), 
                                        None, 
                                        batch_size,shuffle=False)
  
  fe=eval_model(net, test_dataloader)
  
  
  return fe

