import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import random
import numpy as np
from VGG19 import VGG19
from CNN import Classifier
import sys



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Reading data")
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
test_set1 = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_transform)
test_loader1 = DataLoader(test_set1, batch_size=32, shuffle=False)
print("Start predicting")

model = VGG19().to(device)
model.load_state_dict(torch.load('vgg19.model'))
model.eval()
pred_1 = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            pred_1.append(y)

model = Classifier().to(device)
model.load_state_dict(torch.load('CNN.model'))
model.eval()
pred_2 = []
with torch.no_grad():
    for i, data in enumerate(test_loader1):
        test_pred = model(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            pred_2.append(y)

#Ensemble
print("Start Ensemble")
ensemble_labels = np.zeros((3347,))
for i in range(len(test_x)):
    if pred_1[i] == pred_2[i]:
        ensemble_labels[i] = pred_1[i]
    else:
        ensemble_labels[i] = pred_2[i]

ensemble_labels = ensemble_labels.astype(int)
with open(sys.argv[2], 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(ensemble_labels):
    #for i, y in  enumerate(pred_1):
        f.write('{},{}\n'.format(i, y))