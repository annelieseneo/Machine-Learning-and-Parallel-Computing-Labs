# import RF classifier module
from sklearn.ensemble import RandomForestClassifier

# import train test split module
from sklearn.model_selection import train_test_split

# import sklearn's load_breast_cancer dataset
from sklearn.datasets import load_breast_cancer

# import for plotting
import matplotlib.pyplot as plt

# import numpy for working with numbers
import numpy as np

# load dataset
cancer = load_breast_cancer()

# split dataset to train set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                    random_state = 0)

# load RF classifier and instantiate with 50 estimators and same random state as 
    # train test split
forest = RandomForestClassifier(n_estimators = 50, random_state = 0)

# train classifier by fitting training data into the model
forest.fit(X_train,y_train)

# train accuracy
print('Accuracy on the training subset:(:.3f)', format(forest.score(X_train, 
                                                                    y_train)))

# test accuracy
print('Accuracy on the test subset:(:.3f)', format(forest.score(X_test, 
                                                                y_test)))

# 1 - 0.965 = 3.5% (small difference, model is not overfit nor underfit)

# find the number of attributes
n_features = cancer.data.shape[1]
print(n_features)
# 30 features

# creates a horizontal bar graph with one bar for each element at the 
    # specified center location 
# feature importance module provides a better view of feature weight
plt.barh(range(n_features), forest.feature_importances_, align='center')

# use the attribute names of the 30 features as yticks
plt.yticks(np.arange(n_features), cancer.feature_names)

# x and y axis labels
plt.xlabel('Feature Importance')
plt.ylabel('Feature')

# show the plot
plt.show()