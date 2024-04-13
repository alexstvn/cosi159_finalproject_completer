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
model = torchvision.models.vgg19(pretrained=True);
model.classifier = torch.nn.Sequential(*(list(model.classifier)[:-2]));
newmodel = model;
print(newmodel);
feature_vector = np.zeros((2386,4096));
count = 0;
load_dir = "C:/Users/Administrator/Downloads/archive (1)/caltech-101/"
# get the path/directory
folder_dir = load_dir + "Faces";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "Leopards";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "Motorbikes";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "binocular";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "brain";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "camera";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "car_side";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "dollar_bill";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "ferry";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "garfield";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "hedgehog";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "pagoda";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "rhino";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "snoopy";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "stapler";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "stop_sign";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "water_lilly";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "windsor_chair";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "wrench";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);
folder_dir = load_dir + "yin_yang";
for images in os.listdir(folder_dir):
    image = os.path.join(folder_dir, images)
    I = Image.open(image);
    I = transform(I);
    #plt.figure();
    #plt.imshow(I.permute(1, 2, 0));
    output = newmodel(I.unsqueeze(0));
    feature_vector[count,:] = np.squeeze(output.detach().numpy(),0);
    count = count + 1;
    print(count);

feature_vector = np.transpose(feature_vector);
scipy.io.savemat('C:/Users/Administrator/Downloads/resnet50.mat', mdict={'arr': feature_vector})