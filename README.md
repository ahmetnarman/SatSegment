# SatSegment

An exercise for image segmentation to detect buildings on a satellite image that uses a single labeled sattelite picture for training.

## Requirements

* Python3
* PyTorch and torchvision
* Modules specified in the requirements.txt file

CUDA is required for training and testing with a GPU, but the model can run on a CPU as well.

### GPU RAM restriction

For this application, it was assumed that the GPU RAM was not sufficient to support a model that has an input image size larger than 256*256. Thus, the input and output sizes of the model are 256*256.

However, it was assumed that this limitation was only for the model size and the memory was sufficient for a batch larger than one image (otherwise batch size would be chosen as one, making minibatch training irrelevant). Still, in order to reduce the usage of GPU RAM, batch size was chosen relatively small with 5 images per batch. 

The algorithm requires approximately 1GB of GPU RAM to run (~660MB used to store the model and the rest is used for storing the model input, output and gradients).

## Usage

The project includes two scripts that are 'train.py' and 'predict.py'. The 'train.py' script includes a class for model definition and a class for the dataset created using the labeled image. 

Running the training script, a dataset folder is created that includes 256*256 images and ground truth segmentations that are cropped up from the original labeled image and then augmented. If a dataset folder is already created, the code loads the training and testing data from there. Then, the train script defines a model and trains it on the training dataset and saves it. 

The prediction script loads the saved model and runs the test images on the model. Then, it combines the 256*256 images to construct the prediction of the original image. Then it shows the original image, ground truth labels and predictions on a grid of images.
