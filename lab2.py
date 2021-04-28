# import sklearn library
import sklearn

# import sklearn's load_breast_cancer dataset
from sklearn.datasets import load_breast_cancer

# load load_breast_cancer dataset
data = load_breast_cancer()

# print information on load_breast_cancer dataset, including characteristics 
    # such as number of instances and predictive attributes, the class, and 
    # a description of all attributes. Dictionary kets, summary statistics, and 
    # missing attribute values are also displayed
print(data)

# create new variable for the class target names
label_names = data['target_names']

# print class target names
print(label_names)

# create new variable for the class target values
labels = data['target']

# print mapped binary values 0 and 1 of index 0
print(labels[0])
# 0 represents malignant cancer whereas 1 represents benign cancer

# create new variable for the feature name
feature_names = data['feature_names']

# print feature name of index 0
print(feature_names[0])

# create new variable for the feature values
features = data['data']

# print feature values
print(features[0])


from sklearn.model_selection import train_test_split

# randomly split the dataset into the 60% Training set and 40% Test set, using 
    # the list of random numbers starting from 42
train, test, train_labels, test_labels = train_test_split(features, labels,
                                                          test_size = 0.40, 
                                                          random_state = 42)

# import the GaussianNB module
from sklearn.naive_bayes import GaussianNB

# initialize the GaussianNB model
gnb = GaussianNB()

#  train the model by fitting it to the data
model = gnb.fit(train, train_labels)

# evaluate the model by making predictions on test data
preds = gnb.predict(test)

# print predicted values
print(preds)
# series of 0s and 1s are the predicted values for the tumor classes

# import accuracy_score to find model accuracy
from sklearn.metrics import accuracy_score

# print model accuracy by comparing the two arrays
print(accuracy_score(test_labels,preds))
# NaïveBayes classifier is 95.18% accurate