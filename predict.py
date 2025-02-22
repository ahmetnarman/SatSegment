# your implementation goes here
# @author: Ahmet Narman

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from train import SatelliteDataset, Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

im = Image.open('rgb.png').convert('RGB') # The given rgb.png file was in RGBA format
gt = Image.open('gt.png')

# Dataset object is created to load the testing data
dataset = SatelliteDataset(im,gt, load=True)

# The trained model was loaded
# To use the optimal model, load 'optimal_model' instead of 'model' when loading
# The optimal model was trained on a GPU
net = Model()
net.load_state_dict(torch.load('model'))
net.eval()
net = net.to(device)

testSample = dataset.getTestBatch()
with torch.no_grad():
    inputs, labels = testSample['image'].float().to(device), testSample['label'].long().to(device)
    outputs = net(inputs)
    
    # The prediction on the big image was done by putting the predicted patches together
    bigIm = dataset.reconstructRaw(outputs.cpu() if torch.cuda.is_available() else outputs)
    plt.imshow(bigIm)
    plt.title('Predicted image segmentation')