# for working with numbers
import numpy as np

# for plotting graphs
import matplotlib.pyplot as plt

# import package for libraries of basic NN algorithms with flexible network 
    # configurations and learning algorithms
import neurolab as nl

# generate data points
min_val = -30
max_val = 30
num_points = 160
x = np.linspace(min_val, max_val, num_points)
y = 2 * np.square(x) + 8 # generate based on this equation
y /= np.linalg.norm(y)

# reshape data
data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

# visualise input data set
plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data-points')

# NN that consists of more than one layer to extract the underlying patterns in 
    # the training data. Will work like a regressor. 
neural_net = nl.net.newff([[min_val, max_val]], [10, 6, 1])
neural_net
# 2 hidden layers with 10 neurons in the first hidden layer, 6 in the second 
    # hidden layer, and 1 in the output layer

# gradient training algorithm
neural_net.trainf = nl.train.train_gd
neural_net.trainf

# train with goal of learning on the data generated
error = neural_net.train(data, labels, epochs=1000, show=100, goal=0.01)
error

# run on the training data-points
output = neural_net.sim(data)
y_pred = output.reshape(num_points)
y_pred

# visualise
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs') # x axis label
plt.ylabel('Error') # y axis label
plt.title('Training error progress') # title of graph

# plot actual versus predicted output
x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred=neural_net.sim(x_dense.reshape(x_dense.size,1)).reshape(x_dense.size)
plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs predicted') # title of graph
plt.show()