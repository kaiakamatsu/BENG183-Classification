# Machine Learning: Classification  
## By: Kai Akamatsu, Kairi Tanaka, Ashish Dalvi

# HELLO BOYS CITE YOUR SOURCES IN EVERY SECTION YOU COMPLETE !!!!!

# Table of Contents 
- [Introduction](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#introduction)
- [Why Classification?](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#why-classification)
- [What are the types of Classification algorithms?](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#what-are-the-types-of-classification-algorithms)
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
- [Biomedical Applications](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#biomedical-applications)
- [Conclusion](https://github.com/kaiakamatsu/BENG183-Classification/blob/main/Classification.md#conclusion)

# Introduction 
The explosive growth of technology in bioinformatics yields new biological datasets, which presents both unique opportunities and challenges. The accumulation of new sequencing, protein structure, and 3-D genome interaction data has revolutionized our understanding of the human body. Amid this abundance of information, the extraction of meaningful insights relies on effective organization and analysis. This is where classification comes into play, a group of supervised machine learning models that can be used to categorize and decipher patterns in biological data. By leveraging various algorithms, we can find invisible correlations between data sets, predict binding sites, and aid in many other biomedical applications. This paper will navigate the current landscape of classification techniques by exploring different types of machine learning algorithms and discussing their powerful applications to biomedical problems. 

# Why Classification? 


# What are the types of Classification algorithms? 

- Logistic Regression
- Random Forest
- k-Nearest Neighbor
- Neural Network
- Support Vector Machines

> **Here, we will briefly discuss three powerful classification algoriths: k-Nearest Neighbor, Neural Network, and Support Vector Machines.**  
> **To build a stronger intuition about k-Nearest Neighbor and Support Vector Machines, please see their respective sections below.**

## k-Nearest neighbor 
The k-Nearest Neighbor (k-NN) method categorizes a new data point based on the k-nearest neighbors around that data point. For example, if k = 11, we will categorize the new data point based on the majority category of the 11 nearest points near it.  
**STEPS**<br>
**1.** Prepare the data containing various data points with known classes or labels <br>
**2.** This is considered the training process<br>
**3.** Introduce a brand new data point (the unknown vegetable) where you want to assign it to a class based on its characteristics. <br>
**4.** The k-NN algorithm identifies the k number of neighbors near the new data point based on similarity scores (using metrics such as Euclidean or Manhattan distancing). This is the concept of looking at your neighbors vegetables that are already labeled. <br>
**5.** These k neighbors will be considered most similar to the data point, where we can classify.<br>
  
**Note:** It is important to plot the validation curve in order to determine the optimal value for k.<br>
<br>
<div align="center">
<img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9cc3fc86-5e8f-4e73-b4ad-ae0061b10c2b_800x585.gif" width="600" height="400">
</div>
  
From:<br>
[KNN Overview](https://www.ibm.com/topics/knn), [StatsQuest Video](https://www.youtube.com/watch?v=HVXime0nQeI&ab_channel=StatQuestwithJoshStarmer)


## Neural networks
A neural network consists of an input layer, _n_ hidden layers, and an output layer. Each node in the layer have a specific weight and bias attached to them, and all layers are connected. As we train the model, we adjust the weight and bias of nodes, through techniques like gradient descent, so that it results in a more accurate model. 
  
**STEPS**<br>
**1.** Prepare the data and preprocess. <br>
**2.** Choose a type of neural network that suits your needs (i.e. convolutional neural network)  
**3.** Design the architecture of the model. How many layers do we want or type of layer do we want? How many nodes per layer? <br>
**4.** Define the loss function (i.e. cross-entropy loss ), which measures the performance of a classification model whose output is a probability value between 0 and 1. Set up the Optimizer as well (i.e. Adam) <br>
**5.** The optimizer will adjust learning rates for each parameter individually, allowing efficient optimization by accommodating both high and low-gradient parameters. This will be the basis for our learning. <br>
**6.** Feed the training data into the model. Keep an eye on the optimizer in order to minimize loss and update weights to improve accuracy. 
Have proper measure to prevent overfitting <br>
[**Example from TensorFlow:**](https://www.tensorflow.org/tutorials/images/cnn)  
```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```  
**7.** Once the  model is ready, introduce new unknown data and runs it through the model<br>  
**Note:** Use TensorFlow

<div align="center">
<img src="https://github.com/kaiakamatsu/BENG183-Classification/assets/64274901/5aa1041e-064a-4dc7-abf9-517f436ed85b" width="600"
 height="350">
</div>


From: <br>
[Convolutional Neural Networks](https://www.analyticsvidhya.com/blog/2021/07/convolution-neural-network-the-base-for-many-deep-learning-algorithms-cnn-illustrated-by-1-d-ecg-signal-physionet/), [Adam](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/#:~:text=In%20summary%2C%20Adam%20optimizer%20is,weights%20during%20deep%20learning%20training), [Cross Entropy Loss](https://www.v7labs.com/blog/cross-entropy-loss-guide#:~:text=Cross%2Dentropy%20loss%2C%20or%20log,diverges%20from%20the%20actual%20label), [Types of Neural Networks](https://www.mygreatlearning.com/blog/types-of-neural-networks/), [TensorFlow Example](https://www.tensorflow.org/tutorials/images/cnn)

## Support Vector Machines
Support Vector Machines (SVMs) classify data points by finding the most optimal decision boundary. The main aim of SVM to find a hyperplane that maximizes the distance from different categories. SVMs also do not have to be linear, meaning we can incorporate kernel functions, which rely on the process of mapping datasets to a higher dimensions that makes data point separation easier. Overall, SVMs play an important role in separating data for classification purposes.

**STEPS**<br>
**1.** Prepare the data and preprocess <br>
**2.** Choose a kernel (i.e. linear, polynomial, sigmoid, Gaussian Radial Basis Function, etc). Choose the kernelbased on the type of data being used<br>
**3.** Tune the kernel hyper parameters (i.e. gamma RBF) <br>
**4.** Set up the SVM model (i.e. Scikit-learn) <br>
**5.** Train the data using the .fit method 
```
svm_cv.fit(X_train,y_train)
```
**6.** Assess the performance using accuracy, recall, etc as flags. Adjust hyper parameters if unsatisfactory <br>
**7.** Once the  model is ready, introduce new unknown data and runs it through the model<br>

<div align="center">
<img src="https://github.com/kaiakamatsu/BENG183-Classification/assets/64274901/fea2349f-704a-4819-b6d8-c5db1ba8e049" width="600"
 height="300">
</div>
  
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
<img src="https://www.kdnuggets.com/wp-content/uploads/c_distance_metrics_euclidean_manhattan_minkowski_oh_3.png" width="300" height="100">
</div>

**Manhattan Distance Formula:**
<div align="center">
<img src="https://www.kdnuggets.com/wp-content/uploads/c_distance_metrics_euclidean_manhattan_minkowski_oh_5.png" width="280" height="80">
</div>

It is essential to have different colors for labels in order to be able to differentiate.  
### Importance of variable "k"

It is important to choose the most optimal k-value as it will be essential for managing outliers/noise in the overall dataset. When k=1, the model makes predictions based solely on the closest neighbor to a data point. This can cause the model to capture noise or outliers, leading to overfitting because it's overly tailored to the training data. However, if the k value is too high, it can lead to oversimplification of the model and the creation of overly generalized boundaries.  
Although there are various ways to compute the optimal k (i.e. square root k), the most accurate will be the k elbow curve method.
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

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

## Limitations  
### Runtime and Storage  
There are several limitations for the k-nearest neighbor model. The run time growns linearly with the size of the training data, meaning it can be extremely extensive with big data sets. This also means that it will take up a significant amount of memory as we are storing all the points. 
### k-NN features
Some features that could possibly limit the model is the neccesity of the optimal k value. Because this k value looks at all the data points equally when using the elbow method, the lack of weighing means that it is easily influenced by noise compared to other models. 

This also means that when running k-nearest neighbor on an unbalanced data set (lets say there is a significant majority), we can have a biased result. This is due to the fact that the majority can dominate the decision of the model, which causes the minority have a higher chance of being misclassified. 

These limitations show the need for consideration and preprocessing of data before applying the k-NN algorithm. Addressing issues related to weighing of data and reduction of noise can improve the model's effectiveness in real-world applications. Additionally, exploring alternative algorithms or methods that can handle large datasets more efficiently might offer solutions to the challenges k-NN faces.

# Support Vector Machine 

## Intuition/Analogy
Support Vector Machines aim to classify labeled data into two classes. Imagine you are looking at a soccer field with two groups: players wearing red jerseys and players wearing blue jerseys. The SVM finds a hyperplane that separates these two groups of players based on their jersey color. It finds a hyperplane that maximizes the distance between the closest players to the line. These players would be the support vectors and other players would not play a role in the position and orientation of the hyperplane. SVM also allows for misclassifications which is important for generalizing.

## Walk-through/Implementation
Support Vector Machines are implemented by creating a hyperplane to separate the labeled data into two groups. Groups are not always linearly separable so we may need to apply a transformation to make separation possible. This transformation is called a kernel which I discuss further below. Let us know take a look at the code to implement a SVM.

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('dataset.csv')

// spliting the data into X and Y
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

// split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


// scaling our data values to normalize them
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

// use the 'rbf' kernel, a non-linear transformation that helps us fit the SVM
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

// run SVM on test data to see predictions
y_pred = classifier.predict(X_test)

// get accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test,y_pred)

// visualize the data
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

<div align="center">
<img src="https://www.mltut.com/wp-content/uploads/2020/12/svm7.png" width="400
 height="250">
</div>



## Supplements
1. Kernels 
## Limitations

# Biomedical Applications 

Machine learning is transforming biomedical sciences. Novel technologies are producing biological/medical datasets of unprecedented scale and resolution. As bioinformaticians, our role is to leverage this data to yield insights into complex biological systems and/or improve diagnosis and treatment.  
Here, we will discuss three key applications of classification to biomedical sciences.  
> **1. Early Detection / Disease Diagnosis**  
> **2. Disease Monitoring**  
> **3. Functional Genomics**  

## Early Detection / Disease Diagnosis  
Classification algorithms can assist in early detection and disease diagnosis. These machine learning tools have the potential to accelerate decision-making of disease diagnosis by automating visual examinations, perhaps to flag abnormalities. Expediting this process while maintaining accuracy can enable early detection of conditions such as skin cancer, in which early diagnosis improves prognosis for survival tremendously.  

> From: [ML and Medical Diagnosis](https://kili-technology.com/data-labeling/machine-learning/machine-learning-and-medical-diagnosis-an-introduction-to-how-ai-improves-disease-detection)

### Classifying tumors
For example, classification algorithms can be applied to distinguish between **Benign** and **Malignant** tumors. 

<div align="center">
<img src="https://www.verywellhealth.com/thmb/LoHuaSfq_qbKGtSNk3yWBeS0f7s=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/514240-article-img-malignant-vs-benign-tumor2111891f-54cc-47aa-8967-4cd5411fdb2f-5a2848f122fa3a0037c544be.png" 
  width="400 height="250">
</div>

> From: [Malignant vs. Benign Tumors](https://www.verywellhealth.com/what-does-malignant-and-benign-mean-514240#:~:text=benign%20or%20malignant.-,Benign%20tumors%20are%20noncancerous.,malignant%2C%20this%20is%20not%20common.)

Let's explore a scenario in which a healthcare provider wants to classify a tumor based on an image. Based on several visual features of the tumor images, can we accurately catergorize the tumor as benign or malignant? 

[Khairunnahar et al. 2019, *Informatics in Medicine Unlocked*](https://www.sciencedirect.com/science/article/pii/S2352914818301497#sec1) explores this question by implementing a logistic regression model for tumor classification. They utilize a labeled breast cancer diagnosis [dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) with the following features: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave points, Symmetry and Fractal dimension. In their analysis, they create a logistic regression model that can predict tumor malignancy with **97.4249% accuracy**. See ROC curve below: 

<div align="center">
<img src="https://ars.els-cdn.com/content/image/1-s2.0-S2352914818301497-gr5_lrg.jpg" 
  width="400 height="250">
</div>

Next, let's explore a scenario in which a bioinformatician wants to classify a tumor based on the cell's gene expression profiles. The phenotypes of various cells are governed by their gene expression patterns. These unique gene expression profiles can be leveraged to classify cell samples into malignant or benign categories. 

[Divate et al. 2022, *cancers*](https://www.mdpi.com/2072-6694/14/5/1185) leverages a deep neural network to distinguish between 37 cancer types for a total of 2848 samples. 

<div align="center">
<img src="https://www.mdpi.com/cancers/cancers-14-01185/article_deploy/html/images/cancers-14-01185-g002.png" 
  width="400 height="250">
</div>

In this model, they utilize gene expression levels for 13,250 genes as input data to a neural network. Their model achieves a **97.33%** overall accuracy on their testing data. Interestingly, they interpret the layers of the neural network to identify several gene signatures that can be used as biomarkers for determining the cancer's tissue of origin. 

## Disease Monitoring  
Classification algorithms can also improve disease monitoring by categorizing the severity of a condition. Monitoring key clinical variables for a disease and leveraging these machine learning models can potentially inform professionals of the direction of disease progression and help them make clinical decisions. 

[Jovel et al. 2021, *Frontiers in Medicine*](https://www.frontiersin.org/articles/10.3389/fmed.2021.771607/full) applies many of the classification algorithms described in this paper to predict the survival of hepatitis patients based on blood tests that reflect liver function.

<div align="center">
<img src="https://www.frontiersin.org/files/Articles/771607/fmed-08-771607-HTML/image_m/fmed-08-771607-g002.jpg" 
  width="500 height="300">
</div>

Notably, they implemented k-nearest neighbors, random forest, support vector machine, and logistic regression models to perform this classification task. **Figure F** above shows that the logistic regression model outperforms other models by achieving a **90%** accuracy. **Figure A** displays that the k-nearest neighbors algorithm performed best with k = 1-5. Analyzing the importance of every feature in the random forest model, as shown in **figure B**, shows that Albumin, Bilirubin, and Protime levels (all proteins or compound found in the liver) are the most important features in predicting the survival outcome of hepatitis patients. Indeed, plotting the data as a scatter plot with these three measurements on the axes (**figure C**) show that these features separate the two categories of patients in 3-dimensional space. Lastly, **figure D** shows the decision surface used by the logistic regression model, reflecting it's effectiveness in separating patients in the two survival outcomes. 

## Functional Genomics 
Although classification tasks are less common in functional genomics, there are certainly some powerful applications of classification algorithms for genomic annotation. 

[Amariuta et al. 2019, *AJHG*](https://www.cell.com/ajhg/fulltext/S0002-9297(19)30108-9) created a regularized logistic regression model (elastic net) to predict transcription factor occupancy at binding motifs based on local epigenomic features. 

<div align="center">
<img src="https://github.com/kaiakamatsu/BENG183-Classification/assets/108591793/3ff4cee9-0d24-4dbc-bdb1-4c9fa3c0fb55" 
  width="400 height="250">
</div>

Here, ChIP-seq data is used as the cellular truth to train the logistic regression model. Hundreds of local epigenetic features, including open chromatin and H3K4me1, are used as variables to predict the probability of TF occupancy at nuceotide resolution. This models achieves a average AUPRC higher than state-of-the-art approaches to predict TF binding. Integration with genome-wide association study (GWAS) data shows that regulatory elements identified by IMPACT explains a large portion of the heritability of complex diseases such as rheumatoid arthritis. 

# Conclusion 

- Classification is a powerful group of supervised machine learning models that can predict the categorical labels of input objects. Each of these classification algorithms introduced in this paper have their own advantages and limitations. 
- Machine-learning is poised to transform biomedical sciences. We introduced three key applications of classification: disease diagnosis, disease monitoring, and functional genomics.
- The development of novel classification algorithm can potentially accelerate clinical decision-making, democratize disease diagnosis/monitoring, and propel therapeutic development. 

