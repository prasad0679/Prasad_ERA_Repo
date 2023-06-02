# Session 5 Assignment
## Base file ERA_S5.ipynb
### This file consists of the base notebook to train and test the model on MNIST dataset for Digit classification. Below is the code block wise detailed explination 
#### Code Block 1.  
1. Import the necessary Pytorch libraries  
#### Code Block 2. 
1. Check the CUDA availability
2. Copy the model.py into the google colab content
3. Copy the utils.py into the google colab content
4. Import the methods from model.py and utils.py
#### Code Block 3. 
1. Define the data transformations to be performed on training data 
   1.1 Center crop the image to do Data augmentation
   1.2 Resize and perform the Random rotations
   1.3 Standardize and Normalize the data as per overall dataset mean and one std. deviaton
 2. Define the data transformations to be performed on test data 
   2.1 Standardize and Normalize the data as per overall dataset mean and one std. deviaton
#### Code Block 4.  
1. Download the train_data and test_data from MNIST dataset and apply the data transformation
2. Shape of the train_data is "torch.Size([60000, 28, 28])" and test_date shape is "torch.Size([10000, 28, 28]))"
#### Code Block 5.  
1. Define the data loader to load the train_data and test_data
2. Batch Size is defined to load 512 images in each batch. "shuffle" parameter is set to "true"
#### Code Block 6.  
1. This code block loads 12 random images from train_data and plot the same using "matplotlib.pyplot" library
#### Code Block 7.  
1. Code block to define the CNN. This code is modularzed into [model.py](https://github.com/prasad0679/Prasad_ERA_Repo/edit/master/TheSchoolofAI/ERA_S5_Assignment/README.md#modelpy)
#### Code Block 8 and Code Block 9. 
1. Code block to define the methods to train and test the model. This code is modularzed into [utils.py](https://github.com/prasad0679/Prasad_ERA_Repo/edit/master/TheSchoolofAI/ERA_S5_Assignment/README.md#utilspy)
#### Code Block 10. 
##### This code block runs the training epochs and reports the train and test accuracy and loss 
1. Stochastic gradient descent (SGD) optimizer has been used with learning rate = 0.01 
2. Step size of 15 is used to reduce the learning rate by 10% after 15 epochs 
3. Total number of Epochs = 20 
4. Training and Testing methods from [utils.py](https://github.com/prasad0679/Prasad_ERA_Repo/edit/master/TheSchoolofAI/ERA_S5_Assignment/README.md#utilspy) is used to 
   4.1 Train the model 
   4.2 Report the Training accuracy and Training loss 
   4.3 Test the model on Test data and report the Test accuracy and loss 
#### Code Block 11. 
##### This method prints the Training and Testing loss and accuracy.This has been defined in the [utils.py](https://github.com/prasad0679/Prasad_ERA_Repo/edit/master/TheSchoolofAI/ERA_S5_Assignment/README.md#utilspy)
***Maximum test accuracy of 99.29% is achieved in 19th Epoch*** 

## model.py
### This file has 2 methods, "__init__" and "forward" which defines the CNN. Below is the network details and model summary:  
1. **Iput image size : 28x28x1** 
2. **Layer 1: Kernel: 3x3x1,32:** output: 26x26x32, r_out: (28+20-3)/1+1= 26 
3. **Layer 2: Kernel: 3x3x32,64:** output: 24x24x64, r_out: (26+20-3)/1+1= 24 
4. **Max pool Layer1: Kernel: 2x2x64,64:** output: 12x12x64, r_out: (24+20-3)/2+1= 12 
5. **Layer 3: Kernel: 3x3x64,128:** output: 10x10x128, r_out: (12+20-3)/1+1= 10 
6. **Layer 4: Kernel: 3x3x128,256:** output: 8x8x256, r_out: (10+20-3)/1+1= 8 
7. **Max pool2: Kernel: 2x2x256,256:** output: 4x4x256,r_out: (10+20-3)/2+1= 4 
8. **Total paramters in Kernel: 4096:** Flatten to get 50 output channels 
9. **Flatten to get 10 output classes from 50**
### Output of last layer is passed to softmax function
### Following is the Model Summary and details of number of parameters in each layer 

1. Conv2d-1: kernel: 3x3x1,32 : No. of parameters: 288+32 bias = 320
2. Conv2d-2: kernel: 3x3x32,64 : No. of parameters: 18432+64 bias = 18496
3. Conv2d-3: kernel: 3x3x64,128 : No. of parameters: 73728+128 bias = 73,856
4. Conv2d-4: kernel: 3x3x128,256 : No. of parameters: 2,94,912+256 bias = 2,95,168
5. Linear -5: 4096*50 + 50 bias = 204,850
6. Linear -6: 50*10+10 bias = 510 
 
```
---------------------------------------------------------------- 
        Layer (type)               Output Shape         Param # 
   ================================================================ 
            Conv2d-1           [-1, 32, 26, 26]             320 
            Conv2d-2           [-1, 64, 24, 24]          18,496 
            Conv2d-3          [-1, 128, 10, 10]          73,856 
            Conv2d-4            [-1, 256, 8, 8]         295,168 
            Linear-5                   [-1, 50]         204,850 
            Linear-6                   [-1, 10]             510 
================================================================ 
Total params: 593,200 
Trainable params: 593,200 
Non-trainable params: 0 
--------------------------------------------------------------- 
Input size (MB): 0.00 
Forward/backward pass size (MB): 0.67 
Params size (MB): 2.26 
Estimated Total Size (MB): 2.94 
---------------------------------------------------------------- 
``` 

## utils.py 
### This file has 3 methods, "GetCorrectPredCount", "train" and "test" : 
1. ***GetCorrectPredCount :*** This method counts the correct predictions by comparing the model prediction with ground truth and increments the count if the prediction is accurate. 
2. ***train:***  
   2.1 Runs the loop for each batch (512 images from train_data) 
   2.2 Gets the prediction from the model 
   2.3 calculate the loss for each image prediction, performs backpropagation, increments the learning rate as defined for next iteration 
   2.4 Keeps the track of correct predictions by comparing to the ground truth, training accuracy and training loss in each epoch (i.e. for the batch of 512 images) 
3. ***test:*** 
   3.1 Runs the loop for each batch (512 images from train_data) 
   3.2 Gets the prediction from the model 
   3.3 calculate the loss for each image prediction 
   3.4 Keeps the track of correct predictions by comparing to the ground truth, test accuracy and test loss in each epoch (i.e. for the batch of 512 images) 
4. ***printTrainTest_LossAcc*** 
   4.1 This method prints the train_loss, test_loss and train and test accuracy using matplotlib.pyplot library
 


