# Digit-Recognizer
The task is to  apply supervised machine learning methods to the MNIST dataset and achieve at least 90% test accuracy
     
The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

###  K-Nearest Neighbors Classifier (KNN)
kNN is an algorithm which finds a group of k objects in the training set that are closest to the test object and bases the assignment of a label on the predominance of a class in this neighborhood.

To classify an unlabeled image 
* the distance of this image to the labeled objects (training data) is computed
* its k-nearest neighbors are identified and sorted in descending order according to
similarities computed
* the most occurred nearest neighbor in top k neighbors is then assigned as the
class label of the object  
  
#### Evaluation
The database used here has training set of 42,000 samples, and a test set of 28,000 samples.
Here, in the given model, value of k = 3 has been used, which gives an accuracy of 0.968 over the test set.
