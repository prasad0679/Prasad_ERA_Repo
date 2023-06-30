# Session 9 Assignment
## Base file ERA_S9_Assignment_v1.ipynb
### This file consists of the base notebook to train and test the model on CIFAR10 dataset for image classification. Below is the structure

#### dataload.py  
##### Consists of functions to load the CIFAR10 dataset and carry out the transfomrations using "Albumentation" library
1. Draw.plotings: Used to plot the train and test images
2. AlbumentationImageDataset : Main class to initialise the constructor
3. loader :  Class to perform
4. Perform the transforms using Albumentation library
5. Returns the train and test loader iterators.

#### model.py 
1. Define the main model network
2. Function to return the model summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 20, 32, 32]             560
       BatchNorm2d-2           [-1, 20, 32, 32]              40
              ReLU-3           [-1, 20, 32, 32]               0
            Conv2d-4           [-1, 32, 32, 32]           5,792
       BatchNorm2d-5           [-1, 32, 32, 32]              64
              ReLU-6           [-1, 32, 32, 32]               0
            Conv2d-7           [-1, 64, 16, 16]          18,496
       BatchNorm2d-8           [-1, 64, 16, 16]             128
              ReLU-9           [-1, 64, 16, 16]               0
        Dropout2d-10           [-1, 64, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]          73,856
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]           2,432
           Conv2d-15           [-1, 64, 16, 16]           8,256
             ReLU-16           [-1, 64, 16, 16]               0
           Conv2d-17             [-1, 32, 8, 8]          18,464
      BatchNorm2d-18             [-1, 32, 8, 8]              64
             ReLU-19             [-1, 32, 8, 8]               0
        Dropout2d-20             [-1, 32, 8, 8]               0
           Conv2d-21             [-1, 64, 6, 6]          18,496
      BatchNorm2d-22             [-1, 64, 6, 6]             128
             ReLU-23             [-1, 64, 6, 6]               0
           Conv2d-24             [-1, 32, 6, 6]          18,464
      BatchNorm2d-25             [-1, 32, 6, 6]              64
             ReLU-26             [-1, 32, 6, 6]               0
           Conv2d-27             [-1, 32, 3, 3]           9,248
      BatchNorm2d-28             [-1, 32, 3, 3]              64
             ReLU-29             [-1, 32, 3, 3]               0
        Dropout2d-30             [-1, 32, 3, 3]               0
           Conv2d-31             [-1, 16, 3, 3]           4,624
      BatchNorm2d-32             [-1, 16, 3, 3]              32
             ReLU-33             [-1, 16, 3, 3]               0
           Conv2d-34             [-1, 10, 3, 3]           1,450
        AvgPool2d-35             [-1, 10, 1, 1]               0
================================================================
Total params: 180,978
Trainable params: 180,978
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.12
Params size (MB): 0.69
Estimated Total Size (MB): 3.83
----------------------------------------------------------------
```
#### run.py

