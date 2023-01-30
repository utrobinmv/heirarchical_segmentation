#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

#modules
import models
from utils import palette

from src.utils import get_instance
from src.utils import load_parameters
from src.img_utils import center_crop, save_images


config_file = 'config_pascal.json'
model_file = 'saved_models/uppernet_best_model.pth'
img_file = '/mnt/extendet_data/projects/workjob/MILTestTasks-task-heirarchical_segmentation/Pascal-part/JPEGImages/2008_001225.jpg'

config = json.load(open(config_file))
num_classes = 7
device = 'cpu'
model = get_instance(models, 'arch', config, num_classes)

checkpoint = torch.load(model_file, map_location='cpu')

model = load_parameters(model,checkpoint['state_dict'])
model.eval()

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(config['img_norm']['mean'], config['img_norm']['std'])

image = Image.open(img_file).convert('RGB')
#image = center_crop(image, 380, 380)
input = normalize(to_tensor(image)).unsqueeze(0)

with torch.no_grad():
    prediction = model(input.to(device))
    prediction = prediction.squeeze(0).cpu().numpy()
    prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

palette = palette.get_voc_palette(num_classes)

save_images(image, prediction, 'saved_models', img_file, palette)



