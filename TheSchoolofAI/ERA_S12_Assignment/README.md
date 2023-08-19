# Session 12 Assignment
## Base file :: ERA_S12_Assignment_v3.ipynb
### This file consists of the base notebook to train and test the model on CIFAR10 dataset for image classification using Resnet18. Below is the structure. This file imports "models/resnet.py"

### Gradio has been used to create the UI based application. This is also hosted on the huggingface spaces app: https://huggingface.co/spaces/prasadpradhan/ERA_Session12_Assignment/blob/main/README.md


#### dataload.py  
##### Consists of functions to load the CIFAR10 dataset and carry out the transfomrations using "Albumentation" library
1. Draw.plotings: Used to plot the train and test images
2. AlbumentationImageDataset : Main class to initialise the constructor
3. loader :  Class to perform
4. Perform the transforms using Albumentation library
5. Returns the train and test loader iterators.

#### models/resnet.py 
1. Define the main model network
2. Function to return the model summary
```
   | Name              | Type               | Params
----------------------------------------------------------
0  | acc               | MulticlassAccuracy | 0     
1  | preplayer1        | Sequential         | 1.9 K 
2  | convblock1        | Sequential         | 74.1 K
3  | resblock1         | Sequential         | 147 K 
4  | resblock2         | Sequential         | 147 K 
5  | convblock2        | Sequential         | 295 K 
6  | convblock3        | Sequential         | 1.2 M 
7  | resblock3         | Sequential         | 2.4 M 
8  | resblock4         | Sequential         | 2.4 M 
9  | maxpoollayer1     | Sequential         | 0     
10 | ffconnectedlayer1 | Linear             | 5.1 K 
11 | dropout           | Dropout            | 0     
----------------------------------------------------------
6.6 M     Trainable params
0         Non-trainable params
6.6 M     Total params
26.301    Total estimated model params size (MB)

```
#### utils.py
##### This moddule consists of the utility class "Plot" and "plot_metrics" to plot the images
##### Function "get_misclassified_data" is used to prepare the list of misclassified images and "display_gradcam_output" function uses GRADCAM library to explain the model predictions for the misclassified images.
