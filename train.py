# your implementation goes here
# @author: Ahmet Narman

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
import shutil
from PIL import Image



Cuda = torch.cuda.is_available()

# %% Class definitions for the task to make the model easy to use and understand, and to make it 
# easy to maintain. It includes methods and attributes relevant to this task but they can be extended

class SatelliteDataset(torch.utils.data.Dataset):
    """
    The dataset of satellite images and the ground truth segmentations that labels buildings on a pixel level
    """
    def __init__(self, image, gt):
        """
        Arguments:
            image: The raw sattelite image that will be used to generate the dataset (W*H*3)
            gt: The labels of the sattelite image (W*H*3)
        """
        assert image.size == gt.size # The input image and labels must have the same W*H
        
        # Because the resolution of the original image is not divisible by 256, the raw data
        # is resized from 2022*1608 to 2048*1536 = 256x(8*6) for convenience
        
        self.originalRes = image.size # Storing the original aspect ratio to convert it back
        self.rawData = image.resize((2048,1536), Image.ANTIALIAS)
        self.rawLabels = gt.resize((2048,1536), Image.ANTIALIAS)
        self.res = self.rawData.size # The raw image resolution (2048*1536)
        self.modelRes = 256 # Denotes the height and width of the model input (256*256)
        
        self.trainData, self.trainLabels = self.getTrainingData()
        
        self.testData, self.testLabels = self.getTestingData()
        
        self.saveDataset(override=True)
        
    def __getitem__(self, idx):
        """Returns the image and label of specified index from the training dataset"""
        
        return self.trainData[idx], self.trainLabels[idx]
        
    def getTrainingData(self):
        images, labels = self.sliceRaw('train')
        
        # Removing images that are completely black from the training set because they provide no context
        blacklist = [] # indexes of images that are black
        for i in range(len(images)):
            extrema = images[i].convert("L").getextrema()
            if extrema == (0,0):
                blacklist.append(i)
        
        images  = [images[i] for i in range(len(images)) if i not in blacklist]
        labels  = [labels[i] for i in range(len(labels)) if i not in blacklist]
        
        # Extending the training dataset by data augmentation
        images, labels = self.augmentData(images,labels)
        
        return images, labels

    def getTestingData(self):
        
        images, labels = self.sliceRaw('test')
        
        return images, labels
    
    def augmentData(self, images, labels):
        # First step of the augmentation is rotating the images in different angles
        # The buildings on the dataset can have different orientations so this augmentation is very useful
        # Each augmentation type is done on the dataset extended by the previous augmentation
        
        # Images rotated by intervals of 90 degrees (Quadruples the data)
        # More frequent rotations could be done as well but 
        for i in range(len(images)): # For loops are done over indexes so transformation can be done on both images and labels
            for ang in range(90,360,90): # zero degrees is not included as it is already in the array
                images.append(TF.rotate(images[i], ang, resample = Image.BILINEAR, expand=True))
                labels.append(TF.rotate(labels[i], ang, resample = Image.BILINEAR, expand=True))
        
        # Images flipped horizontally (Doubles the data)
        for i in range(len(images)):
            images.append(TF.hflip(images[i]))
            labels.append(TF.hflip(labels[i]))
            
        # Images flipped vertically (Doubles the data)        
        for i in range(len(images)):
            images.append(TF.vflip(images[i]))
            labels.append(TF.vflip(labels[i]))

        # Brightness and contrast adjusted images added (Triples the data)
        # They could be adjusted separately but that would make the dataset too big
        for i in range(len(images)):
            images.append(TF.adjust_contrast(TF.adjust_brightness(images[i], brightness_factor=1.2), contrast_factor=1.2))
            labels.append(labels[i]) # Label brightness does not change
            images.append(TF.adjust_contrast(TF.adjust_brightness(images[i], brightness_factor=0.8), contrast_factor=0.8))
            labels.append(labels[i]) # Label brightness does not change
              
        return images, labels
    
    def sliceRaw(self, mode):
        """Slices the raw data (image and ground truth) into 256*256 pieces"""
        if mode == 'test':  
            # For testing data, only the images that are just sliced from the raw image are used
            images = []
            labels = []
            temp = self.modelRes
            for i in range(int(self.res[0]/temp)):
                for j in range(int(self.res[1]/temp)):
                    # Slicing the raw data into square pieces
                    images.append(TF.crop(self.rawData, j*temp, i*temp, temp, temp))
                    labels.append(TF.crop(self.rawLabels, j*temp, i*temp, temp, temp))         
            return images, labels
        else:
            # For training data, the croppings are done with shorter intervals to increase the amount of data used
            images = []
            labels = []
            temp = self.modelRes/2 # To reduce the cropping interval from 256 to 128
            for i in range(int(self.res[0]/temp)-1):
                for j in range(int(self.res[1]/temp)-1):
                    # Slicing data into overlapping square pieces
                    images.append(TF.crop(self.rawData, j*temp, i*temp, temp*2, temp*2))
                    labels.append(TF.crop(self.rawLabels, j*temp, i*temp, temp*2, temp*2))         
            return images, labels

    def saveDataset(self, override=False):
        """
        Saves the images and labels extracted from the raw images to local files
        
        """
        if not os.path.isdir('dataset') or override:
            
            if os.path.isdir('dataset'):
                shutil.rmtree('dataset') # shutil is used to remove a nonempty directory
            os.mkdir('dataset')
            
            os.chdir('dataset')
            os.mkdir('train')
            os.mkdir('test')
            
            os.chdir('train')
            os.mkdir('images')
            os.mkdir('labels')
  
            os.chdir('..')
            os.chdir('test')     
            os.mkdir('images')
            os.mkdir('labels')
            
            os.chdir('..')
            os.chdir('..')
            
            for i in range(len(self.trainData)):
                self.trainData[i].save(os.path.join('dataset', 'train', 'images', 'im'+str(i)+'.png'))
                self.trainLabels[i].save(os.path.join('dataset', 'train', 'labels', 'gt'+str(i)+'.png'))
            
            for i in range(len(self.testData)):
                self.testData[i].save(os.path.join('dataset', 'test', 'images', 'im'+str(i)+'.png'))
                self.testLabels[i].save(os.path.join('dataset', 'test', 'labels', 'gt'+str(i)+'.png'))
        else:
            print('The dataset is already created. To save a new one, use the override option')

            
class SatelliteDataLoader(torch.utils.data.DataLoader):
    
    def __init__(self, batch_sie=1, shuffle=False):
        pass
        # I dont know about this one
        #super().__init__(super().batch_size=batch_size, super().shuffle=shuffle)
        
class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 2, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.interpolate(F.relu(self.conv3(x)), scale_factor=2, mode='bilinear') # TODO check if this layer works properly
        x = self.conv4(x)
        return x

#%% Loading, visualizing and analyzing the data

# It was found that the input PNG file has W*H*4 dimensions, which is caused by the image being in RGBA format
# For this practice, we are only interested in RGB dimensions
        
im = Image.open('rgb.png').convert('RGB')
gt = Image.open('gt.png')

imDim = im.size # 2022*1608

dataset = SatelliteDataset(im, gt)

trainIms = dataset.trainData
trainLabels = dataset.trainLabels

testIms = dataset.testData
testLabels = dataset.testLabels
"""
plt.figure()
for i in range(8):
    rInd = np.random.randint(0,high = len(testIms))
    plt.suptitle('Data samples from the testing dataset')
    plt.subplot(4,8,i+1)
    plt.imshow(testIms[rInd])
    plt.subplot(4,8,i+9)
    plt.imshow(testLabels[rInd])
    rInd = np.random.randint(0,high = len(testIms))
    plt.subplot(4,8,i+17)
    plt.imshow(testIms[rInd])
    plt.subplot(4,8,i+25)
    plt.imshow(testLabels[rInd])
"""
plt.figure()
for i in range(8):
    plt.suptitle('Data samples from the training dataset')
    rInd = np.random.randint(0,high = len(trainIms))
    plt.subplot(4,8,i+1)
    plt.imshow(trainIms[rInd])
    plt.subplot(4,8,i+9)
    plt.imshow(trainLabels[rInd])
    rInd = np.random.randint(0,high = len(trainIms))
    plt.subplot(4,8,i+17)
    plt.imshow(trainIms[rInd])
    plt.subplot(4,8,i+25)
    plt.imshow(trainLabels[rInd])
plt.show()


# It was found that the ground truth contains two values for every pixel: 
# - (255,255,255): White segments in the ground truth image
# - (0,0,0): Black segments in the ground truth image

plt.figure(figsize=(13,6))
plt.subplot(1,2,1)
plt.imshow(dataset.rawData)
plt.title('Input image')
plt.subplot(1,2,2)
plt.imshow(dataset.rawLabels)
plt.title('Ground truth labels')


#%% Data preprocessing



        
# Because of the memory limitations, the input size of our model can be 256*256*3 at most
# Thus, the input image and the corresponding ground truth images should be separated into smaller images
# for training and evaluating.

# Because PyTorch batch notation has the format of [batch_size, channels, height, width], the W*H*3 array
# notation will be rearranged for tensors

# Normalization is done on the input data


'''
imPart = imNp[800:918,0:128,:]
gtPart = gtNp[800:918,0:128,:]

plt.figure(figsize=(13,6))
plt.subplot(1,2,1)
plt.imshow(imPart)
plt.title('Input image')
plt.subplot(1,2,2)
plt.imshow(gtPart)
plt.title('Ground truth labels')
'''