# Session 8 Assignment
## Base file ERA_Session8_Assignment_BN_v1.ipynb --> Using the BatchNorm
## Base file ERA_Session8_Assignment_LayerNorm_v1 --> using the LayerNorm
## Base file ERA_Session8_Assignment_GroupNorm_v1 --> using the GroupNorm
### This file consists of the model with 48,736 parameters and last 3 epoch achieving accuracy > 70%

## model_bn.py --> Batch Normalization model summary
 
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           9,216
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             512
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11           [-1, 32, 16, 16]           4,608
             ReLU-12           [-1, 32, 16, 16]               0
      BatchNorm2d-13           [-1, 32, 16, 16]              64
          Dropout-14           [-1, 32, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           9,216
             ReLU-16           [-1, 32, 16, 16]               0
      BatchNorm2d-17           [-1, 32, 16, 16]              64
          Dropout-18           [-1, 32, 16, 16]               0
           Conv2d-19           [-1, 16, 16, 16]             512
        MaxPool2d-20             [-1, 16, 8, 8]               0
           Conv2d-21             [-1, 32, 8, 8]           4,608
             ReLU-22             [-1, 32, 8, 8]               0
      BatchNorm2d-23             [-1, 32, 8, 8]              64
          Dropout-24             [-1, 32, 8, 8]               0
           Conv2d-25             [-1, 32, 6, 6]           9,216
             ReLU-26             [-1, 32, 6, 6]               0
      BatchNorm2d-27             [-1, 32, 6, 6]              64
          Dropout-28             [-1, 32, 6, 6]               0
           Conv2d-29             [-1, 32, 4, 4]           9,216
             ReLU-30             [-1, 32, 4, 4]               0
      BatchNorm2d-31             [-1, 32, 4, 4]              64
          Dropout-32             [-1, 32, 4, 4]               0
        AvgPool2d-33             [-1, 32, 1, 1]               0
           Conv2d-34             [-1, 10, 1, 1]             320
================================================================
Total params: 48,736
Trainable params: 48,736
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.81
Params size (MB): 0.19
Estimated Total Size (MB): 3.01
---------------------------------------------------------------- 
``` 
## model_ln.py --> Layer Normalization model summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
         GroupNorm-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           9,216
              ReLU-6           [-1, 32, 32, 32]               0
         GroupNorm-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             512
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11           [-1, 32, 16, 16]           4,608
             ReLU-12           [-1, 32, 16, 16]               0
        GroupNorm-13           [-1, 32, 16, 16]              64
          Dropout-14           [-1, 32, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           9,216
             ReLU-16           [-1, 32, 16, 16]               0
        GroupNorm-17           [-1, 32, 16, 16]              64
          Dropout-18           [-1, 32, 16, 16]               0
           Conv2d-19           [-1, 16, 16, 16]             512
        MaxPool2d-20             [-1, 16, 8, 8]               0
           Conv2d-21             [-1, 32, 8, 8]           4,608
             ReLU-22             [-1, 32, 8, 8]               0
        GroupNorm-23             [-1, 32, 8, 8]              64
          Dropout-24             [-1, 32, 8, 8]               0
           Conv2d-25             [-1, 32, 6, 6]           9,216
             ReLU-26             [-1, 32, 6, 6]               0
        GroupNorm-27             [-1, 32, 6, 6]              64
          Dropout-28             [-1, 32, 6, 6]               0
           Conv2d-29             [-1, 32, 4, 4]           9,216
             ReLU-30             [-1, 32, 4, 4]               0
        GroupNorm-31             [-1, 32, 4, 4]              64
          Dropout-32             [-1, 32, 4, 4]               0
        AvgPool2d-33             [-1, 32, 1, 1]               0
           Conv2d-34             [-1, 10, 1, 1]             320
================================================================
Total params: 48,736
Trainable params: 48,736
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.81
Params size (MB): 0.19
Estimated Total Size (MB): 3.01
----------------------------------------------------------------
```
## model_gn.py --> Group Normalization model summary : Group size = 2
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
         GroupNorm-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           9,216
              ReLU-6           [-1, 32, 32, 32]               0
         GroupNorm-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             512
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11           [-1, 32, 16, 16]           4,608
             ReLU-12           [-1, 32, 16, 16]               0
        GroupNorm-13           [-1, 32, 16, 16]              64
          Dropout-14           [-1, 32, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           9,216
             ReLU-16           [-1, 32, 16, 16]               0
        GroupNorm-17           [-1, 32, 16, 16]              64
          Dropout-18           [-1, 32, 16, 16]               0
           Conv2d-19           [-1, 16, 16, 16]             512
        MaxPool2d-20             [-1, 16, 8, 8]               0
           Conv2d-21             [-1, 32, 8, 8]           4,608
             ReLU-22             [-1, 32, 8, 8]               0
        GroupNorm-23             [-1, 32, 8, 8]              64
          Dropout-24             [-1, 32, 8, 8]               0
           Conv2d-25             [-1, 32, 6, 6]           9,216
             ReLU-26             [-1, 32, 6, 6]               0
        GroupNorm-27             [-1, 32, 6, 6]              64
          Dropout-28             [-1, 32, 6, 6]               0
           Conv2d-29             [-1, 32, 4, 4]           9,216
             ReLU-30             [-1, 32, 4, 4]               0
        GroupNorm-31             [-1, 32, 4, 4]              64
          Dropout-32             [-1, 32, 4, 4]               0
        AvgPool2d-33             [-1, 32, 1, 1]               0
           Conv2d-34             [-1, 10, 1, 1]             320
================================================================
Total params: 48,736
Trainable params: 48,736
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.81
Params size (MB): 0.19
Estimated Total Size (MB): 3.01
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


