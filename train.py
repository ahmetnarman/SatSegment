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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SatelliteDataset(torch.utils.data.Dataset):
    """
    The dataset of satellite images and the ground truth segmentations that labels buildings on a pixel level
    This dataset is created using the raw image and the ground truth label provided
    The dataset is created or loaded from local files if it is already created
    """
    def __init__(self, image, gt, load=True):
        """
        Arguments:
            image: The raw sattelite image (PIL) that will be used to generate the dataset (W*H*3)
            gt: The labels of the sattelite image (PIL) (W*H*3)
        """
        assert image.size == gt.size # The input image and labels must have the same W*H
        
        # Because the resolution of the original image is not divisible by 256, the raw data
        # is resized from 2022*1608 to 2048*1536 = 256x(8*6) for convenience
        
        self.originalRes = image.size # Storing the original aspect ratio to convert it back
        self.rawData = image.resize((2048,1536), Image.ANTIALIAS)
        self.rawLabels = gt.resize((2048,1536), Image.ANTIALIAS)
        self.res = self.rawData.size # The raw image resolution (2048*1536)
        self.modelRes = 256 # Denotes the height and width of the model input (256*256)
        print(os.path.isdir('dataset'))
        if load and os.path.isdir('dataset'):
            self.trainData,self.trainData,self.testData,self.testLabels = self.loadDataset()
        else:
            self.createDataset()
            print('New dataset created successfully...')
            
            self.trainData,self.trainLabels,self.testData,self.testLabels = self.loadDataset()
        
    def __len__(self):
        """Returns the size of the training dataset"""
        return len(self.trainData)
        
    def __getitem__(self, idx):
        """
        Returns the image and label of specified index from the training dataset in tensor form
        Used by the dataLoader object of PyTorch
        """
        
        image,label = self.getSample(idx)
        imTensor, gtTensor = self.im2tensor(image, label) 
        
        sample = {'image':imTensor, 'label':gtTensor}
        return sample
    
    def loadDataset(self):
        """ Loads the list of images and labels, for training and testing sets """
        
        trData = os.listdir(os.path.join('dataset','train','images'))
        trLabels = os.listdir(os.path.join('dataset','train','labels'))
        
        teData = os.listdir(os.path.join('dataset','test','images'))
        teLabels = os.listdir(os.path.join('dataset','test','labels'))
        
        return trData, trLabels, teData, teLabels
    
    def getTestBatch(self):
        """
        This function returns the testing data in a batch for putting through the module
        as the dataLoader only provides samples from the training set
        """
        
        images = []
        labels = []
        for i in range(len(os.listdir(os.path.join('dataset','test','images')))):
            images.append(np.array(Image.open(os.path.join('dataset','test','images','im'+str(i)+'.png'))).transpose((2,0,1)))
            labels.append(np.array(Image.open(os.path.join('dataset','test','labels','gt'+str(i)+'.png'))).transpose((2,0,1))[0,:,:]/255)
        
        imTensor = torch.from_numpy(np.stack(images, axis=0)) # Stacking to create a batch
        gtTensor = torch.from_numpy(np.stack(labels, axis=0))
                
        sample = {'image':imTensor, 'label':gtTensor}
        
        return sample
    
    def getSample(self, idx):
        """ Returns an image,label pair from local files in PIL image format """
        
        image = Image.open(os.path.join('dataset','train','images','im'+str(idx)+'.png'))
        label = Image.open(os.path.join('dataset','train','labels','gt'+str(idx)+'.png'))
        
        return image,label
    
    def im2tensor(self, image, label):
        """ 
        Converts images to tensors to be fed through the model
        Because PyTorch batch notation has the format of [batch_size, channels, height, width], 
        the W*H*3 array notation will be rearranged for the tensors
        """
        
        imTensor = torch.from_numpy(np.array(image).transpose((2,0,1))) # Rearranging the dimensions into 3*W*H
        
        # Below tensor has dimensions of W*H. Pixel values: 1=background, 0=building
        gtTensor = torch.from_numpy(np.array(label).transpose((2,0,1))[0,:,:]/255)
        
        return imTensor.float(), gtTensor.long() # Converting to the required format for training
        
    def onehot2im(self, labTensor):
        """ 
        Converts tensors produced by the model to images
        For visualizing the segmentations
        The input to this function should be an image tensor of shape 2*W*H instead of a batched tensor
        """
        
        label = np.array(torch.argmax(labTensor, 0), dtype='uint8')*255
        label = np.stack((label,label,label),axis=2)
        label = Image.fromarray(label)
        return label
    
    def reconstructRaw(self, output):
        """ Takes in the batched testing output of the neural net and reconstructs the segmentation of the original RGB image"""
        
        images = []
        for i in range(output.shape[0]):
            images.append(np.array(self.onehot2im(output[i,:,:,:]), dtype='uint8'))
        
        bigImage = np.zeros((self.res[1],self.res[0],3))
        ind = 0
        for i in range(int(self.res[0]/self.modelRes)):
            for j in range(int(self.res[1]/self.modelRes)):
                bigImage[j*self.modelRes:(j+1)*self.modelRes,i*self.modelRes:(i+1)*self.modelRes,:] = images[ind]
                ind += 1
        return bigImage        
        
    def buildTrainingData(self):
        """Generates and returns the training images and labels, uses data augmentation"""
        
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

    def buildTestingData(self):
        """Generates and returns the testing images and labels"""
        
        images, labels = self.sliceRaw('test')
        
        return images, labels
        
    def augmentData(self, images, labels):
        """
        Applies different augmentation techniques to the data to extend the training dataset and improve generalization performance
        """
        
        # Images rotated by intervals of 90 degrees (Quadruples the data)
        # More frequent rotations could be done as well but that makes the training dataset too large
        for i in range(len(images)):
            for ang in range(90,360,90): # zero degrees is not included as it is already in the array
                images.append(TF.rotate(images[i], ang, resample = Image.BILINEAR, expand=True))
                labels.append(TF.rotate(labels[i], ang, resample = Image.BILINEAR, expand=True))
        
        # At the end of the loop, the original unrotated images are removed because they are also in the testing set
        # Training the model using the data you use for testing is a No-No
        del images[:i+1]
        del labels[:i+1]
        
        # Images flipped horizontally (Doubles the data)
        for i in range(len(images)):
            images.append(TF.hflip(images[i]))
            labels.append(TF.hflip(labels[i]))
            
        # Images flipped vertically (Doubles the data)
        for i in range(len(images)):
            images.append(TF.vflip(images[i]))
            labels.append(TF.vflip(labels[i]))
        
        # Brightness and contrast adjusted images added (Triples the data)
        # They could be adjusted separately but that would make the dataset too large
        for i in range(len(images)):
            images.append(TF.adjust_contrast(TF.adjust_brightness(images[i], brightness_factor=1.2), contrast_factor=1.2))
            labels.append(labels[i]) # Label brightness does not change
            images.append(TF.adjust_contrast(TF.adjust_brightness(images[i], brightness_factor=0.8), contrast_factor=0.8))
            labels.append(labels[i]) # Label brightness does not change
        
        # With current augmentation pipeline, around 5600 training images are generated
        
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

    def createDataset(self):
        """
        Saves the images and labels extracted from the raw images intoto local files
        WARNING: If you already have a dataset created, this method overrides it
        """

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
        
        trainIms,trainLabs = self.buildTrainingData()
        testIms,testLabs = self.buildTestingData()
        
        for i in range(len(trainIms)):
            trainIms[i].save(os.path.join('dataset', 'train', 'images', 'im'+str(i)+'.png'))
            trainLabs[i].save(os.path.join('dataset', 'train', 'labels', 'gt'+str(i)+'.png'))
        
        for i in range(len(testIms)):
            testIms[i].save(os.path.join('dataset', 'test', 'images', 'im'+str(i)+'.png'))
            testLabs[i].save(os.path.join('dataset', 'test', 'labels', 'gt'+str(i)+'.png'))
        
        
class Model(nn.Module):
    """Neural network model definition"""
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 2, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x))) # Downsampling by max pooling
        x = F.interpolate(F.relu(self.conv3(x)), scale_factor=2, mode='bilinear') # Upsampling by interpolation
        x = self.conv4(x)
        return x


### THE MAIN TRAINING SCRIPT STARTS HERE ###
        
def main():       
    
    im = Image.open('rgb.png').convert('RGB') # The given rgb.png file was in RGBA format
    gt = Image.open('gt.png')
    
    imDim = im.size # 2022*1608
    
    dataset = SatelliteDataset(im, gt, load=True)
    batch_size = 5
    
    # DataLoader to be used in training, batch size defined here, shuffling enabled to improve training performance
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=2)
    
    # Checking if the dataloader returns the batches with proper size
    for i_batch, sample_batched in enumerate(trainLoader):
        print(i_batch, sample_batched['image'].size(),sample_batched['label'].size())
        if i_batch == 3:
            break
    
    
    """
    # Plotting the raw image and ground truth
    plt.figure(figsize=(13,6))
    plt.subplot(1,2,1)
    plt.imshow(dataset.rawData)
    plt.title('Input image')
    plt.subplot(1,2,2)
    plt.imshow(dataset.rawLabels)
    plt.title('Ground truth labels')
    plt.show()
    
    
    # Plotting random images from the training set
    plt.figure()
    for i in range(8):
        plt.suptitle('Data samples from the training dataset')
        rInd = np.random.randint(0,high = len(dataset.trainData))
        image, label = dataset.getSample(rInd)
        plt.subplot(4,8,i+1)
        plt.imshow(image)
        plt.subplot(4,8,i+9)
        plt.imshow(label)
        rInd = np.random.randint(0,high = len(dataset.trainData))
        image, label = dataset.getSample(rInd)
        plt.subplot(4,8,i+17)
        plt.imshow(image)
        plt.subplot(4,8,i+25)
        plt.imshow(label)
    plt.show()
    """
    
    # TODO Normalization is done on the input data
    
    # Model definition, (It requires approximately 660MB of GPU RAM)
    net = Model().to(device) # If CUDA is available, the model will use GPU for training and testing
    
    # As each pixel is classified into two classes, cross-entrophy loss function was used
    optimizerLoss = nn.CrossEntropyLoss()
    
    #Adam was chosen as optimizer because of its superior optimization performance
    optimizer = torch.optim.Adam(net.parameters(), betas=(0.9, 0.99)) 
    
    it = 1 # The number of iterations
    losses = []
    epochs = 40
    
    # When the batch size is 5, training does not require more than 1GB of RAM
    for epoch in range(epochs):
        epochLoss = 0
        for i_batch, sample_batched in enumerate(trainLoader):
            #print('Batch: {}'.format(i_batch), end='\r')
            
            inputs, labels = sample_batched['image'].to(device), sample_batched['label'].to(device)
            
            # making parameter gradients zero before forward and back pass
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = optimizerLoss(outputs, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
            
            it += 1
        
        # Run through the test samples at the end of each epoch
        testSample = dataset.getTestBatch()
        # Because the testing batch is big, it requires like 1GB of RAM on its own
        with torch.no_grad():
            inputs, labels = testSample['image'].float().to(device), testSample['label'].long().to(device)
            
            outputs = net(inputs)
            bigIm = dataset.reconstructRaw(outputs.cpu())
            #plt.figure()
            plt.imshow(bigIm)
            loss = optimizerLoss(outputs, labels)
            print(loss)
    
    torch.save(net.state_dict(), 'model')
    
if __name__ == "__main__":main()