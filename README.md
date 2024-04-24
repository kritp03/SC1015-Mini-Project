# SC1015 Mini Project: Facial Emotion Recognition

![alt text](https://github.com/kritp03/SC1015-Mini-Project/blob/main/assets/cover.jpeg)

## About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on emotions from FER2013 dataset and FERPLUS2013 dataset

![alt text](https://github.com/kritp03/SC1015-Mini-Project/blob/main/assets/emotion_image.png)

## Problem Statement 

Understanding and managing students' emotional responses in educational settings is crucial in improving learning outcomes, especially for children who are unable to express themselves well verbally. We plan to create a deep learning model which is capable of accurately categorizing facial expressions into eight emotional states using convolutional neural networks (CNNs) trained on datasets like FER-2013 and FERPLUS. By discerning students' emotional reactions, educators can personalize teaching approaches and support methods to cater to individual learning needs effectively. 

This brings us to our problem statement: How can we optimize children's learning experience by analyzing their emotional response during task engagement?

## Models Used and Model Performances
    1) CNN Model
    2) CNN + Batch Normalization + Dropout
    3) CNN + Batch Normalization + Dropout + Data Augmentation
    4) ResNet-50

| Model | Train Accuracy | Validate Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| Baseline CNN | 0.9921  | 0.7247 | 0.6227 |
| CNN with BN + Dropout | 0.9721  | 0.7686 | 0.7518 |
| CNN with BN + Dropout + Data Augmentation | 0.8996  | 0.8179 | 0.8167 |
| ResNet-50 | 0.8760  | 0.8761 | 0.8752 |
   
## Conclusion
<b>Best Performance Model </b>: CNN + Batch Normalization + Dropout + Data Augmentation (3rd Model)

The model with Batch Normalization, Dropout and Data Augmentation performed the best in terms of metrics and image predictions. It attained an accuracy of 0.8167, and a loss of 0.1483, outperforming the primitive CNN model. 

Our ResNet-50 model attained an accuracy of 0.8752 but a loss of 1.5536. Despite having a higher accuracy than the tuned CNN, it was unable to perform well on unseen data due to `Domain Shift`, `Feature Relevance` as well as `Bias and Generalization`. More of this is elaborated in our notebook 

The baseline CNN model achieved an accuracy of 0.6227, and a loss of 7.1700 while the tuned CNN model (with Batch Normalization, Dropout and Data Augmentation) achieved an accuracy of 0.8142, and a loss of 0.1441.

<b>Summary of Performance Metrics</b>

<table>
  <tr>
    <th>Model</th>
    <th>Train Accuracy</th>
    <th>Test Accuracy</th>
    <th>Validate Accuracy</th>
  </tr>
  <tr>
    <td>Baseline CNN</td>
    <td>0.9921</td>
    <td>0.6227</td>
    <td>0.7247</td>
  </tr>
  <tr>
    <td>CNN with Batch Normalization (BN) + Dropout</td>
    <td>0.9721</td>
    <td>0.7518 </td>
    <td>0.7686</td>
  </tr>
  <tr>
    <td>CNN with BN + Dropout + Data Augmentation</td>
    <td>0.8996</td>
    <td>0.8167</td>
    <td>0.8179</td>
  </tr>
  <tr>
    <td>Pretrained ResNet-50 </td>
    <td>0.8760</td>
    <td>0.8752</td>
    <td>0.8761</td>
  </tr>
</table>


<h3><b>Solution to Problem Statement: </b></h3>

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

## Future Work
- `Tackle Imbalanced datasets`: Leverage on methods like `re-weighting` with the sklearn class_weight.compute_class_weight function. This strategy allows us to modify our loss function by assigning `increased costs` to instances from `minority` classes, offering the potential to enhance the model's efficacy on datasets with imbalanced class distributions.


## References
1. [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013/data)
2. [FERPLUS Dataset](https://www.kaggle.com/datasets/ss1033741293/ferplus)
3. [ResNet](https://towardsdatascience.com/resnets-why-do-they-perform-better-than-classic-convnets-conceptual-analysis-6a9c82e06e53)
4. [Addressing Class Imbalance](https://medium.com/@dudjakmario/addressing-the-problem-of-class-imbalance-part-1-4-9690d9cd41a2)
5. [Batch Normalization](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/#:~:text=Batch%20normalization%20works%20by%20normalizing,not%20follow%20the%20original%20distribution.)
6. [Early Stopping](https://paperswithcode.com/method/early-stopping#:~:text=Early%20Stopping%20is%20a%20regularization,improves%20on%20a%20validation%20set.)
7. [Dropout](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)

## Contributors
| Name | GitHub Account |
| --- | --- |
| Sih Jia Qi | [@kritp03](https://github.com/kritp03) |
| Tee Wei Ping | [@sihjiaqi](https://github.com/sihjiaqi) |
| Ponyuenyong Kritchanat | [@weipingtee](https://github.com/weipingtee) |
