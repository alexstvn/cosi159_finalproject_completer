# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:25:30 2024

@author: Administrator
"""
import numpy as np
import scipy.io
from PIL import Image
from scipy import ndimage, misc

import os
from os import listdir
import os
import time
import timeit
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import argparse

import torch
import torchvision
from torchvision import transforms
class AverageMeter:
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
            
# model
transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor()
    ])
model = torchvision.models.resnet50(pretrained=True)
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
print(newmodel)
feature_vector = np.zeros((4485,2048))
count = 0
# get the path/directory
folder_name = "./Desktop/CompleterTest/data/15-Scene/C"
# Data was rearranged to match the order of the existing Scene-15 dataset for COMPLETER
for f in range(9):
    folder_dir = folder_name + f"0{f+1}"
    for images in os.listdir(folder_dir):
        image = os.path.join(folder_dir, images)
        I = Image.open(image)
        I = transform(I)
        #plt.figure();
        #plt.imshow(I.permute(1, 2, 0));
        output = newmodel(I.unsqueeze(0))
        feature_vector[count,:] = np.squeeze(output[0,:,0].detach().numpy(),1)
        count = count + 1
        print(count)
for f in range(6):
    folder_dir = folder_name + f"{f+10}"
    for images in os.listdir(folder_dir):
        image = os.path.join(folder_dir, images)
        I = Image.open(image)
        I = transform(I)
        #plt.figure();
        #plt.imshow(I.permute(1, 2, 0));
        output = newmodel(I.unsqueeze(0))
        feature_vector[count,:] = np.squeeze(output[0,:,0].detach().numpy(),1)
        count = count + 1
        print(count)

#feature_vector = np.transpose(feature_vector)
scipy.io.savemat('./Desktop/CompleterTest/data/sceneresnet50.mat', mdict={'arr': feature_vector})