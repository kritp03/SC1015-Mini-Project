# SC1015 Mini Project: Facial Emotion Recognition

## About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on emotions from  <a href="">FER2013 dataset</a> and <a href="">FERPLUS2013 dataset</a>

## Problem Statement 
- How can we optimize children's learning experience by analyzing their emotional response during task engagement?

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
Technical:
    1. Overfitting: 
     - When the training data size is too small and lacks samples to accurately represent all input values, it can lead to overfitting as the model is unable to generalize and fits too closely to the training dataset.
     - Finding the 'sweet spot' of the learning rate is important as the extreme ends can lead to overfitting and underfitting.
    2. 
Ethical:

Data Analysis:

## References
https://towardsdatascience.com/resnets-why-do-they-perform-better-than-classic-convnets-conceptual-analysis-6a9c82e06e53

## Contributors
    - @kritp03:
    - @sihjiaqi: 
    - @weipingtee