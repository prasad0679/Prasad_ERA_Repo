# Session 6 Assignment
## Base file ERA_S6_Assignment_v3.ipynb
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
2. Batch Size is defined to load 128 images in each batch. "shuffle" parameter is set to "true"
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
***Maximum test accuracy of 99.50% is achieved in 18th and 20th Epoch*** 

## model.py
### This file has 2 methods, "__init__" and "forward" which defines the CNN. Below is the model summary:  
 
```
<add the momdel summary> 
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


