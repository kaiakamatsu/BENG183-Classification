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
In the field of bioinformatics, with the explosive growth of technology also came the increase in biological data, which presents both unique opportunities and challenges. Through the accumulation of new sequences, protein structures, and much more, this data has revolutionized the understanding of the human body. Amid this abundance of information, the extraction of meaningful insights crucially relies on effective organization and analysis. This is where classification comes into play, a tool for categorizing and deciphering biological data patterns by using supervised machine learning. By leveraging various algorithms, we can find invisible correlations between data sets, predicting binding sites, and aiding in many other biomedical applications. This paper will navigate the landscape of classification techniques from the bottom up, by exploring different types of machine learning algorithms to biological data analysis, where we emphasize the power of classification in bioinformatics. 
# Why Classification? 

# What are the types of Classification algorithms? 
## k-Nearest neighbor 
The k-Nearest Neighbor (k-NN) method is like asking your closest neighbors for advice. Imagine you're trying to figure out what type of an unknown vegetable is in your basket. You look at the labeled vegetables your neighbors have, compare them, and decide based on the most similar ones.
  
**STEPS**<br>
1 Prepare the data containing various data points with known classes or labels <br>
2 This is considered the training process<br>
3 Introduce a brand new data point (the unknown vegetable) where you want to assign it to a class based on its characteristics. <br>
4 The k-NN algorithm identifies the k number of neighbors near the new data point based on similarity scores (using metrics such as Euclidean or Manhattan distancing). This is the concept of looking at your neighbors vegetables that are already labeled. <br>
5 These k neighbors will be considered most similar to the data point, where we can classify.<br>
  
**Note:** It is important to plot the validation curve in order to determine the optimal value for k.<br>
<br>
  
![k nearest neighbors visualization](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9cc3fc86-5e8f-4e73-b4ad-ae0061b10c2b_800x585.gif)

From:<br>
[KNN Overview](https://www.ibm.com/topics/knn), [StatsQuest Video](https://www.youtube.com/watch?v=HVXime0nQeI&ab_channel=StatQuestwithJoshStarmer)


## Neural networks
A neural network is like a team learning to identify fruits. Each member specializes in recognizing specific fruit traits (color, shape, texture) just as neural network layers focus on different aspects. They share insights like how neurons exchange information. Training involves showing labeled fruit examples, improving their ability to identify fruits. Testing checks their proficiency. The team gets better with more fruit exposure, similar to how neural networks improve with more data.<br>
  
**STEPS**<br>
1 Prepare the data and preprocess. <br>
2 Choose a type of neural network that suits your needs (i.e. convolutional neural network) <br>
3 Design the architecture of the model. How many layers do we want or type of layer do we want? How many nodes per layer? <br>
4 Define the loss function (i.e. cross-entropy loss ), which measures the performance of a classification model whose output is a probability value between 0 and 1. Set up the Optimizer as well (i.e. Adam) <br>
5 The optimizer will adjust learning rates for each parameter individually, allowing efficient optimization by accommodating both high and low-gradient parameters. This will be the basis for our learning. <br>
6 Feed the training data into the model. Keep an eye on the optimizer in order to minimize loss and update weights to improve accuracy. 
Have proper measure to prevent overfitting <br>
7 Once the  model is ready, introduce new unknown data and runs it through the model<br>
  
![999181_BIpRgx5FsEMhr1k2EqBKFg (1)](https://github.com/kaiakamatsu/BENG183-Classification/assets/64274901/5aa1041e-064a-4dc7-abf9-517f436ed85b)

From: <br>
[Convolutional Neural Networks](https://www.analyticsvidhya.com/blog/2021/07/convolution-neural-network-the-base-for-many-deep-learning-algorithms-cnn-illustrated-by-1-d-ecg-signal-physionet/), [Adam](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/#:~:text=In%20summary%2C%20Adam%20optimizer%20is,weights%20during%20deep%20learning%20training), [Cross Entropy Loss](https://www.v7labs.com/blog/cross-entropy-loss-guide#:~:text=Cross%2Dentropy%20loss%2C%20or%20log,diverges%20from%20the%20actual%20label), [Types of Neural Networks](https://www.mygreatlearning.com/blog/types-of-neural-networks/)

## Support Vector Machines
Support Vector Machines (SVMs) are similar to a wise judge creating a clear boundary in a courtroom. Picture a judge presiding over a complex case with two sides presenting different arguments. The judge (SVM) seeks a fair and decisive line (decision boundary) that separates the two stances, ensuring a just verdict. Just as the judge carefully weighs evidence and arguments to establish a balanced judgment, SVMs analyze data to create an optimal boundary that maximizes the margin between classes, ensuring a clear distinction between different data groups, much like the judge's ruling brings clarity to a legal dispute.<br>
  
**STEPS**<br>
1 Prepare the data and preprocess <br>
2 Choose a kernel (i.e. linear, polynomial, sigmoid, Gaussian Radial Basis Function, etc). Choose the kernelbased on the type of data being used<br>
3 Tune the kernel hyper parameters (i.e. gamma RBF) <br>
4 Set up the SVM model (i.e. Scikit-learn) <br>
5 Train the data using the .fit method 
```
svm_cv.fit(X_train,y_train)
```
6 Assess the performance using accuracy, recall, etc as flags. Adjust hyper parameters if unsatisfactory <br>
7 Once the  model is ready, introduce new unknown data and runs it through the model<br>
  
![SVM visualization](https://github.com/kaiakamatsu/BENG183-Classification/assets/64274901/fea2349f-704a-4819-b6d8-c5db1ba8e049)

From:  
[SVM Overview](https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/#h-kernels-in-support-vector-machine), [StatsQuest Video](https://www.youtube.com/watch?v=efR1C6CvhmE&ab_channel=StatQuestwithJoshStarmer)
# K-Nearest Neighbors Classification 

## Intuition/Analogy
As stated before, the k-Nearest Neighbor (k-NN) method is like asking your closest neighbors for advice. Imagine you're trying to figure out what type of an unknown vegetable is in your basket. You look at the labeled vegetables your neighbors have, compare them, and decide based on the most similar ones.
## Walk-through/Implementation 
### Training  
The training process of the KNN classification algorithm is fairly straightfoward. The algorithm stores the entire data set to memory while keeping track of distances (for example Euclidean Distance).  
In python we can calculate these using **numpy** or **scipy**  
**Euclidean Distance Formula:**  
<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9LeaMTcOXxeTPN-VCbKloQ.png" width="300" height="100">
</div>

**Manhattan Distance Formula:**
<div align="center">
<img src="https://www.kdnuggets.com/wp-content/uploads/c_distance_metrics_euclidean_manhattan_minkowski_oh_5.png" width="300" height="100">
</div>

It is essential to have different colors for labels in order to be able to differentiate.  
### Importance of variable "k"

It is important to choose the most optimal k-value as it will be essential for managing outliers/noise in the overall dataset. When k=1, the model makes predictions based solely on the closest neighbor to a data point. This can cause the model to capture noise or outliers, leading to overfitting because it's overly tailored to the training data. However, if the k value is too high, it can lead to oversimplification of the model and the creation of overly generalized boundaries.  
Although there are various ways to compute the optimal k (i.e. square root k), the most accurate will be the k elbow curve method.
```
data = pd.read_csv(‘KNN_Project_Data) \\ read data

error_rate = [] \\keep track of error rates through various k values 
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6)) //plot example 
 // ... include color implementations, etc. 
plt.title(‘Error Rate vs. K Value’)
plt.xlabel(‘K’)
plt.ylabel(‘Error Rate’)
```
By keeping track of the error rates and the k values, we will be looking at which k value has the lowest error rate. By doing so we can prevent over/under fitting.  
For example:  

<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:1316/format:webp/1*DevIKIpXaJm7xkodmcefQg.png" width="400
 height="250">
</div>
  
The optimal k value will be 20 as it has the lowest error rate.   
[Elbow Method](https://medium.com/@moussadoumbia_90919/elbow-method-in-supervised-learning-optimal-k-value-99d425f229e7)
  
### Introduction of a New Data Point  

<div align="center">
<img src="https://github.com/kaiakamatsu/BENG183-Classification/assets/64274901/253a59b4-6810-4eee-930c-7fe859800781" width="400
 height="250">
</div>

[Image Credits](https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning)  

When we introduce a new data point we will calculate the distances from the new data point to the others using the chosen distance metric. 
Then we can sort the calculated distances in ascending order based on distance values. This way, we can look at the top k rows for the closest k neighbors. Ultimately, we can see which neighbors the new data point is and make a decision on which label it belongs due to the highest vote. 



## Supplements
1. How is distance defined?
   The distance between points u and v. We can use various distance metrics as shown above. 
3. How does K affect classification?
## Limitations

# Support Vector Machine 

## Intuition/Analogy
## Walk-through
## Implementation
## Supplements
1. Kernels 
## Limitations

# Biological Applications 

## Types

# Conclusion 
