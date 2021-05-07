# import pandas package
import pandas as pd

# import numpy package for statistics
import numpy as np

# import svm module for svc classification, and datasets to load iris data, 
    # from sklearn library
from sklearn import svm, datasets

# import package to plot graphs
import matplotlib.pyplot as plt

# load input data
iris = datasets.load_iris()

# take the first two features/attributes
X = iris.data[:, :2]

# print values of first two features/attributes
print(X)

# target class labels
y = iris.target

# print class labels values
print(y)

# plot the support vector machine boundaries with original data
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
print(h)

# create a mesh to plot
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

X_plot = np.c_[xx.ravel(), yy.ravel()]
print(X_plot)

# value of regularization parameter
C = 1.0 

# call classifier for training and creating SVM classifier object
# use linear kernel function
# X is attributes, y is labels
svc_classifier = svm.SVC(kernel = 'linear', C = C, 
                         decision_function_shape = 'ovr').fit(X, y)

# evaluate model performance by predicting test data
Z = svc_classifier.predict(X_plot)

# print predictings in an array
print(Z)

# reshape the array to meshgrid
Z = Z.reshape(xx.shape)
print(Z)

# plotting graph

# plot size
plt.figure(figsize=(15, 5))
plt.subplot(121)

# add countour lines to current plot, and set background colour according to 
    # the contour found for the 3 classes based on dataset points
plt.contourf(xx, yy, Z, cmap = plt.cm.tab10, alpha = 0.3)

# add points of specified colours
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1)

# x and y axis labels
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# x axis limits
plt.xlim(xx.min(), xx.max())

# graph title
plt.title('SVC with linear kernel')