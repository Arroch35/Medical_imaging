#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:00:29 2022

@author: francesco
"""
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_pretrained_vit import ViT
from transformers import AutoImageProcessor, ViTModel
from torchvision import transforms
import os

from timm.layers import set_fused_attn
set_fused_attn(False)
# uni_model_pth=r'D:\Experiments\DigiPatics\PearsonColon\Code\FeatExtraction\uni_model\ckpts'

uni_model_pth=os.path.join(os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1]), 'uni_model','ckpts')
from uni_model.get_encoder import get_encoder


class FeatureExtractorVGG(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()
        
        if model_name == 'Vgg16':
            net = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
        elif model_name == 'Vgg19':
            net = models.vgg19(weights = models.VGG19_Weights.DEFAULT)
        
        if cut_index is not None:
            for i in range(cut_index,len(net.classifier)):
               net.classifier[i] = nn.Identity()
                
        self.fe = net

    def forward(self, x):
        x = self.fe(x)
                
        return x


class FeatureExtractorResnet152(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT) #pretrained = True)
        net.fc = nn.Identity()

        self.fe = net

    def forward(self, x):
        x = self.fe(x)

        return x

class FeatureExtractorResnet18(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Identity()

        self.fe = net

    def forward(self, x):
        x = self.fe(x)

        return x

class FeatureExtractorDENSENET169(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        net.classifier = nn.Identity()

        self.fe = net

    def forward(self, x):
        x = self.fe(x)

        return x


class FeatureExtractorEFFICIENTNETB7(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        net.classifier[0] = nn.Identity()
        net.classifier[1] = nn.Identity()

        self.fe = net

    def forward(self, x):
        x = self.fe(x)

        return x
        
class FeatureExtractorMOBILENETV2(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        #for param in net.features.parameters():  --> això és el que feia el guillermo
        #    param.requires_grad = False
        net.classifier[0] = nn.Identity()
        net.classifier[1] = nn.Identity()

        self.fe = net

    def forward(self, x):
        x = self.fe(x)

        return x
        
class FeatureExtractorEFFICIENTNETV2s(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = models.efficientnet_v2_s(pretrained = True)
        net.classifier[0] = nn.Identity()
        net.classifier[1] = nn.Identity()

        self.fe = net

    def forward(self, x):
        x = self.fe(x)

        return x

#### VISUAL TRANSFORMERS
class FeatureExtractorGoogleViT(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()
        
        self.fe = ViTModel.from_pretrained("google/vit-base-patch16-224")

    def forward(self, x):
        
        out = self.fe(x, output_attentions=True) 
        # feat = out.last_hidden_state  # drop CLS => (196,768)
        # layers = [att.mean(dim=1)[1:,1:] for att in out.attentions]
        
        
        return out.last_hidden_state, out.attentions[-1]


class FeatureExtractorViTB16(nn.Module):
    def __init__(self, model_name, image_size=None,cut_index=None):
        super().__init__()
        
        
        if model_name == 'ViTB16':
            net = ViT('B_16_imagenet1k', pretrained=True,image_size=image_size)
        elif model_name == 'ViTB32':
            net = ViT('B_32_imagenet1k', pretrained=True,image_size=image_size)
        elif model_name == 'ViT32':
           net = ViT('B_32', pretrained=True,image_size=image_size)
        elif model_name == 'ViT16':
              net = ViT('B_16', pretrained=True,image_size=image_size)
     
        
        net.fc = nn.Identity()
        
        self.fe = net

    def forward(self, x):
        x = self.fe(x)
        scores=self.fe.transformer.blocks[-1].attn.scores
        
        return x,scores

class FeatureExtractorViTB16Attention(nn.Module):
    def __init__(self, model_name, image_size=None,cut_index=None):
        super().__init__()
        
        
        if model_name == 'ViTB16':
            net = ViT('B_16_imagenet1k', pretrained=True,image_size=image_size)
        elif model_name == 'ViTB32':
            net = ViT('B_32_imagenet1k', pretrained=True,image_size=image_size)
        elif model_name == 'ViT32':
           net = ViT('B_32', pretrained=True,image_size=image_size)
        elif model_name == 'ViT16':
              net = ViT('B_16', pretrained=True,image_size=image_size)
     
        
        net.fc = nn.Identity()
        
        self.fe = net

    def forward(self, x):
        x = self.fe(x)
        scores=self.fe.transformer.blocks[-1].attn.scores
        
        return x,scores

class FeatureExtractorViTB16Patches(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = ViT('B_16_imagenet1k', pretrained=True)
        net.fc = nn.Identity()
        
        self.fe = net

    def forward(self, x,mask=None):
        
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.fe.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
    #    x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        x = self.fe.transformer(x,mask=mask)  # b,gh*gw+1,d
       # x = self.norm(x)[:, 1::]  # b,gh*gw,d
        x = self.fe.norm(x)  # b,gh*gw+1,d
        scores=self.fe.transformer.blocks[-1].attn.scores
        
        return x,scores

class FeatureExtractorViTB32Patches(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = ViT('B_32_imagenet1k', pretrained=True)
        net.fc = nn.Identity()
        
        self.fe = net

    def forward(self, x,mask=None):
        
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.fe.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
    #    x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        x = self.fe.transformer(x,mask=mask)  # b,gh*gw+1,d
       # x = self.norm(x)[:, 1::]  # b,gh*gw,d
        x = self.fe.norm(x)  # b,gh*gw+1,d
        scores=self.fe.transformer.blocks[-1].attn.scores
        
        return x,scores

class FeatureExtractorUNIViT(nn.Module):
    def __init__(self):
        super().__init__()
        net, transform = get_encoder(enc_name="uni2-h",assets_dir=uni_model_pth  )
               
        self.fe = net
        self.transform=transform
    def forward(self, x,mask=None):
        # x[:,0,:] is cls global context; x[:,1:net.num_reg_tokens::,:]  are the register tokens; 
        # x[:,net.num_reg_tokens+1::,:] are the patch features
        x=self.fe.forward_features(x) 
        scores=self.fe.blocks[-1].attn.attn
        
        return x,scores
        