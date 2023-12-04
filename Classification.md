# Machine Learning: Classification  
## By: Kai Akamatsu, Kairi Tanaka, Asish Dalvi

# HELLO BOYS CITE YOUR SOURCES IN EVERY SECTION YOU COMPLETE !!!!!

# Table of Contents 
- [Introduction](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#introduction)
- [Why Classification?](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#why-classification)
- [What are the types of Classification algorithms?](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#what-are-the-types-of-classification-algorithms)
  - [Which algorithm should be used?](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#which-algorithm-should-be-used
)
- [K-Nearest Neighbors Classification](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#k-nearest-neighbors-classification)
  - [Intuition/Analogy](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#intuitionanalogy)
  - [Walk-through](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#walk-through)
  - [Implementation](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#implementation)
  - [Supplements](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#supplements)
- [Support Vector Machine](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#support-vector-machine)
  - [Intuition/Analogy](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#intuitionanalogy-1)
  - [Walk-through](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#walk-through-1)
  - [Implementation](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#implementation-1)
  - [Supplements](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#supplements-1)
- [Biological Applications](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#biological-applications)
- [Conclusion](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#conclusion)

# Introduction 

# Why Classification? 

# What are the types of Classification algorithms? 
## k-Nearest neighbor 
The k-Nearest Neighbor (k-NN) method is like asking your closest neighbors for advice. Imagine you're trying to figure out what type of an unknown vegetable is in your basket. You look at the labeled vegetables your neighbors have, compare them, and decide based on the most similar ones.

With the k-NN method you are: 
Prepare the data containing various data points with known classes or labels 
This is considered the training process
Introduce a brand new data point (the unknown vegetable) where you want to assign it to a class based on its characteristics. 
The k-NN algorithm identifies the k number of neighbors near the new data point based on similarity scores (using metrics such as Euclidean or Manhattan distancing). This is the concept of looking at your neighbors vegetables that are already labeled. 
These k neighbors will be considered most similar to the data point, where we can classify.

It is important to plot the validation curve in order to determine the optimal value for k.

![k nearest neighbors visualization](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9cc3fc86-5e8f-4e73-b4ad-ae0061b10c2b_800x585.gif)

## Neural networks
Supervised learning with neural networks is like training a talented apprentice. Imagine teaching a skilled apprentice artisan the art of pottery. You provide examples of various pottery shapes (input data) and explain the desired shapes (labels or outputs). The apprentice (neural network) learns by observing these examples and adjusting their techniques (weights and biases) to replicate the desired pottery shapes (predictions) when creating new pottery pieces (unseen data). Over time, with continuous practice and guidance (training iterations), the apprentice becomes adept at crafting pottery that closely resembles the desired shapes, demonstrating the ability to generalize and create new pieces (make predictions) based on the learned patterns from your teachings (training data).

Prepare the data and preprocess. 
Choose a type of neural network that suits your needs (i.e. convolutional neural network) 
Design the architecture of the model. How many layers do we want or type of layer do we want? How many nodes per layer? 
 Define the loss function (i.e. cross-entropy loss ), which measures the performance of a classification model whose output is a probability value between 0 and 1. Set up the Optimizer as well (i.e. Adam) 
The optimizer will adjust learning rates for each parameter individually, allowing efficient optimization by accommodating both high and low-gradient parameters. This will be the basis for our learning. 
Feed the training data into the model. Keep an eye on the optimizer in order to minimize loss and update weights to improve accuracy. 
Have proper measure to prevent overfitting 
Once the  model is ready, introduce new unknown data and runs it through the model

![999181_BIpRgx5FsEMhr1k2EqBKFg (1)](https://github.com/kaiakamatsu/BENG183-Classification/assets/64274901/5aa1041e-064a-4dc7-abf9-517f436ed85b)


# K-Nearest Neighbors Classification 

## Intuition/Analogy
## Walk-through
## Implementation
## Supplements
1. How is distance defined?
2. How does K affect classification?

# Support Vector Machine 

## Intuition/Analogy
## Walk-through
## Implementation
## Supplements

# Biological Applications 

# Conclusion 
