# SatSegment

## GPU RAM restriction

For this application, it was assumed that the GPU RAM was not sufficient to support a model that has an input image size larger than 256*256. However, it was assumed that this limitation was only for the model size and the memory was sufficient for a batch larger than one image (otherwise batch size would be chosen as one, making minibatch training irrelevant). At a given time only a batch of images and ground truth labels are kept in GPU memory (not the whole dataset).
