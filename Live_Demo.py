# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 21:22:52 2022

@author: PC
"""

import torch
import torchvision

model = torchvision.models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

model.load_state_dict(torch.load('best_model.pth'))

device = torch.device('cuda')
model = model.to(device)

import cv2
import numpy as np

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

import traitlets
from jetbot import Camera, bgr8_to_jpeg

camera = Camera.instance(width=224, height=224)

from jetbot import Robot

robot = Robot()

import torch.nn.functional as F
import time

def update(change):
    global blocked_slider, robot
    x = change['new'] 
    x = preprocess(x)
    y = model(x)
    
    # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
    y = F.softmax(y, dim=1)
    prob_blocked = float(y.flatten()[0])
    
    if prob_blocked < 0.5:
        robot.forward(0.4)
    else:
        robot.left(0.4)
    
    time.sleep(0.001)
        
update({'new': camera.value})

camera.observe(update, names='value')

import time

camera.unobserve(update, names='value')

time.sleep(0.1)  # add a small sleep to make sure frames have finished processing

robot.stop()

camera_link.unlink()