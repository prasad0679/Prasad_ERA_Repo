# Session 7 Assignment
## Base file ERA_S7_Assignment_v5.ipynb
### This file consists of the model with 10,812 parameters and last 3 epoch achieving accuracy > 99.40%

## model.py
### This file has 2 methods, "__init__" and "forward" which defines the CNN. Below is the model summary:  
 
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 26, 26]              40
              ReLU-2            [-1, 4, 26, 26]               0
       BatchNorm2d-3            [-1, 4, 26, 26]               8
           Dropout-4            [-1, 4, 26, 26]               0
            Conv2d-5            [-1, 8, 24, 24]             296
              ReLU-6            [-1, 8, 24, 24]               0
       BatchNorm2d-7            [-1, 8, 24, 24]              16
           Dropout-8            [-1, 8, 24, 24]               0
         MaxPool2d-9            [-1, 8, 12, 12]               0
           Conv2d-10            [-1, 8, 12, 12]              72
             ReLU-11            [-1, 8, 12, 12]               0
      BatchNorm2d-12            [-1, 8, 12, 12]              16
          Dropout-13            [-1, 8, 12, 12]               0
           Conv2d-14           [-1, 16, 10, 10]           1,168
             ReLU-15           [-1, 16, 10, 10]               0
      BatchNorm2d-16           [-1, 16, 10, 10]              32
          Dropout-17           [-1, 16, 10, 10]               0
           Conv2d-18             [-1, 32, 8, 8]           4,640
             ReLU-19             [-1, 32, 8, 8]               0
      BatchNorm2d-20             [-1, 32, 8, 8]              64
          Dropout-21             [-1, 32, 8, 8]               0
           Conv2d-22             [-1, 16, 8, 8]             528
           Conv2d-23             [-1, 16, 6, 6]           2,320
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
           Conv2d-27             [-1, 10, 4, 4]           1,450
             ReLU-28             [-1, 10, 4, 4]               0
      BatchNorm2d-29             [-1, 10, 4, 4]              20
          Dropout-30             [-1, 10, 4, 4]               0
           Conv2d-31             [-1, 10, 4, 4]             110
        AvgPool2d-32             [-1, 10, 1, 1]               0
================================================================
Total params: 10,812
Trainable params: 10,812
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.41
Params size (MB): 0.04
Estimated Total Size (MB): 0.45
---------------------------------------------------------------- 
``` 

## utils.py 
### This file has 3 methods, "GetCorrectPredCount", "train" and "test" : 
1. **GetCorrectPredCount :** This method counts the correct predictions by comparing the model prediction with ground truth and increments the count if the prediction is accurate. 
2. **train:**  
   2.1 Runs the loop for each batch (512 images from train_data)  
   2.2 Gets the prediction from the model  
   2.3 calculate the loss for each image prediction, performs backpropagation, increments the learning rate as defined for next iteration  
   2.4 Keeps the track of correct predictions by comparing to the ground truth, training accuracy and training loss in each epoch (i.e. for the batch of 512 images)  
3. **test:**  
   3.1 Runs the loop for each batch (512 images from train_data)  
   3.2 Gets the prediction from the model  
   3.3 calculate the loss for each image prediction  
   3.4 Keeps the track of correct predictions by comparing to the ground truth, test accuracy and test loss in each epoch (i.e. for the batch of 512 images)  
4. **printTrainTest_LossAcc**  
   4.1 This method prints the train_loss, test_loss and train and test accuracy using matplotlib.pyplot library.  


