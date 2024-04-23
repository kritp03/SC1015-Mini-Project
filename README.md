# SC1015 Mini Project: Facial Emotion Recognition

## About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on emotions from  <a href="">FER2013 dataset</a> and <a href="">FERPLUS2013 dataset</a>

## Problem Statement 

Understanding and managing students' emotional responses in educational settings is crucial in improving learning outcomes, especially for children who are unable to express themselves well verbally. We plan to create a deep learning model which is capable of accurately categorizing facial expressions into eight emotional states using convolutional neural networks (CNNs) trained on datasets like FER-2013 and FERPLUS. By discerning students' emotional reactions, educators can personalize teaching approaches and support methods to cater to individual learning needs effectively. This brings us to our problem statement: How can we optimize children's learning experience by analyzing their emotional response during task engagement?


## Models used
    1) CNN Model
    2) CNN + Batch Normalization + Dropout
    3) CNN + Batch Normalization + Dropout + Data Augmentation
    4) ResNet-50
   
## Conclusion
Best Performance Model: ResNet-50

The ResNet-50 model performed the best with an accuracy of 0.8752, and a loss of 1.5336, outperforming the primitive CNN model. ResNets are able to perform better than the general CNN models as they are built by stacking residual blocks on top of one another and go as long as a hundred layers per network, efficiently learning all the parameters from early activations deeper in the network. 

The baseline CNN model achieved an accuracy of 0.6227, and a loss of 7.1700 while the tuned CNN model (with Batch Normalization, Dropout and Data Augmentation) achieved an accuracy of 0.8142, and a loss of 0.1441.

Solution to Problem Statement:

Using our model to detect emotions in children's faces can help us understand their emotional response during task engagement. This project has inspired the potential for technology to enhance education, especially in the field of early childhood education where children may not always be able to verbalize their thoughts. The introduction of this technology can aid educators in understanding their students better through their facial expressions.

## Learning Points

    1. Overfitting: 
    
     - When the training data size is too small and lacks samples to accurately represent all input values, it can lead to overfitting as the model is unable to generalize and fits too closely to the training dataset.
     
     - Finding the 'sweet spot' of the learning rate is important as the extreme ends can lead to overfitting and underfitting.


    2. Data Analysis

    - Feature Engineering: to introduce a new column which stores the emotion with highest number of votes
    - Visualizations using Plotly as it is interactive, clean and easy to understand.
    - Data Augmentation: process of creating additional image samples from the original dataset, can help represent underrepresentated classes and make the dataset more robust and balaned


    3. Modelling

    - Used ResNet-50 to train on top of the basic CNN models as pre-trained models have already undergone extensive training, so we can achieve impressive results without having to access expensive hardware or massive datasets. 

    - To be mindful of the number of layers that we are adding: stacking layers may not necessarily be a good thing as adding layers increases the number of weights in the network and the model complexity. If we do not have a large dataset, an increasingly large network can result in overfitting.

    - Dropout layers: Improves the model's generalization performance by preventing overfitting. Randomly deactivating neurons during training prevents co-adaptation, and this forces the model to find and use truly robust features in the data. 

    - Early Stopping: Stops training when parameter updates no longer begin to yield improves on a validation set. The use of early stopping prevent overfitting & enhance generalization. 

    - Batch Normalization: Helps address issues like internal covariate shift and vanishing/exploding gradients which leads to faster convergence during training, allowing for the use of higher learning rates and reducing the sensitivity of the model to weight initialization


## References
https://towardsdatascience.com/resnets-why-do-they-perform-better-than-classic-convnets-conceptual-analysis-6a9c82e06e53

https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/#:~:text=Batch%20normalization%20works%20by%20normalizing,not%20follow%20the%20original%20distribution.

https://paperswithcode.com/method/early-stopping#:~:text=Early%20Stopping%20is%20a%20regularization,improves%20on%20a%20validation%20set.

https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/

## Contributors
@kritp03:

@sihjiaqi: 

@weipingtee

<a href="https://github.com/OWNER/REPO/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OWNER/REPO" />
</a>
