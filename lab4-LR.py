# import numpy package for statistics
import numpy as np

# import svm module for LR classification from sklearn library
from sklearn import linear_model 

# import package to plot graphs
import matplotlib.pyplot as plt

# define sample data

# features/attributes
X = np.array([[2, 4.8], [2.9, 4.7], [2.5, 5], [3.2, 5.5], [6, 5], [7.6, 4], 
              [3.2, 0.9], [2.9, 1.9], [2.4, 3.5], [0.5, 3.4], [1, 4], [0.9, 5.9]])

# 4 target class labels
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# load and initialise the LR classifier
Classifier_LR = linear_model.LogisticRegression(solver = 'liblinear', C = 75)

# train classifier
Classifier_LR.fit(X, y)

# used to visualise output
def Logistic_visualize(Classifier_LR, X, y):
    
    #  defined the minimum and maximum values X and Y to be used in mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
 
    # define the step size for plotting the mesh grid
    mesh_step_size = 0.02
 
    # define the mesh grid of X and Y values 
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), 
                                 np.arange(min_y, max_y, mesh_step_size))
    
    # run the classifier on the mesh grid
    # evaluate model performance by making predictions
    output = Classifier_LR.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    print(output)
    
    # reshape the array to meshgrid
    output = output.reshape(x_vals.shape)
    print(output)
    
    # plotting graph
    plt.figure()
    
    # add countour lines to current plot, and set background colour according to 
        # the contour found for the 4 classes based on dataset points
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    
    # add points of specified colours
    plt.scatter(X[:, 0], X[:, 1], c = y, s = 75, edgecolors = 'black', 
                linewidth = 1, cmap = plt.cm.Paired)
    
    # specify the boundaries of the plot
    
    # x and y axis limits
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    
    # x and y axis ticks
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))
    
    # display plot
    plt.show()

# visualise output of LR
Logistic_visualize(Classifier_LR, X, y)