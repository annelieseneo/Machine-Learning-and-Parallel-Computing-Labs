# import sklearn library
import sklearn

# import sklearn's load_breast_cancer dataset
from sklearn.datasets import load_breast_cancer

# load dataset into data variable
data = load_breast_cancer()

# create new variable for the class target names
label_names = data['target_names']

# print types of labels (2 classes of serious and not serious classes)
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

# print feature values of attribute 1
print(features[0])

# import train test split module
from sklearn.model_selection import train_test_split

# randomly split dataset to 60% train set and 40% test set
# random_state will assign same random set of numbers starting from position 42
    # as seed. model is stabilised
# parameters/arguments : features and labels are both split
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size = 0.40, 
                                                          random_state = 42)

# import Gaussian NB algorithm from NB class
from sklearn.naive_bayes import GaussianNB

# load and initialise Gaussion NB module, without arguments
gnb = GaussianNB()

# train classifier using train set by fitting as model object
model = gnb.fit(train, train_labels)

# predict target class values of test set to evaluate the model
preds = gnb.predict(test)

# print predicted values
print(preds)
# series of 0s and 1s are the predicted values for the tumor classes


# import accuracy score module
from sklearn.metrics import accuracy_score

# print accuracy score of model, by comparing arrays of actual labels vs pred values
print(accuracy_score(test_labels, preds))
# Gaussian NB model is 95.18% accurate