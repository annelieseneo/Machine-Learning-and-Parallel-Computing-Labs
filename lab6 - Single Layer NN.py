# for working with numbers
import numpy as np

# for plotting graphs
import matplotlib.pyplot as plt

# import package for libraries of basic NN algorithms with flexible network 
    # configurations and learning algorithms
import neurolab as nl

# load dataset from txt file, first 2 columns are attributes, last 2 are labels
input_data = np.loadtxt(r"C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\neural_simple.txt")

# load dataset       
# =============================================================================
# array([[2. , 4. , 0. , 0. ],
#      [1.5, 3.9, 0. , 0. ],
#      [2.2, 4.1, 0. , 0. ],
#      [1.9, 4.7, 0. , 0. ],
#      [5.4, 2.2, 0. , 1. ],
#      [4.3, 7.1, 0. , 1. ],
#      [5.8, 4.9, 0. , 1. ],
#      [6.5, 3.2, 0. , 1. ],
#      [3. , 2. , 1. , 0. ],
#      [2.5, 0.5, 1. , 0. ],
#      [3.5, 2.1, 1. , 0. ],
#      [2.9, 0.3, 1. , 0. ],
#      [6.5, 8.3, 1. , 1. ],
#      [3.2, 6.2, 1. , 1. ],
#      [4.9, 7.8, 1. , 1. ],
#      [2.1, 4.8, 1. , 1. ]])
# =============================================================================

# seperate into 2 data columns and 2 labels
data = input_data[:, 0:2]
labels = input_data[:, 2:]

# plot input data
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1') # x axis label
plt.ylabel('Dimension 2') # y axis label
plt.title('Input data') # title of plot

# min and max values of column dimension 0 and 1, for theta
dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()

# define number of neurons in output layer
nn_output_layer = labels.shape[1]
nn_output_layer

# NN with independent neurons acting on input data to produce output
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
neural_net = nl.net.newp([dim1, dim2], nn_output_layer) # define network, input & output
neural_net

# train NN, input is data, output is labels
error = neural_net.train(data, labels, epochs=200, show=20, lr=0.01)
error

# visualise training progress
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs') # x axis label
plt.ylabel('Training error') # y axis label
plt.title('Training error progress') # title of plot
plt.grid()
plt.show()

# test data
data_test = [[1.5, 3.2], [3.6, 1.7], [3.6, 5.7],[1.6, 3.9]] 

# print test results
print('\nTest Results:')
for item in data_test:
    print(item, '-->', neural_net.sim([item])[0])