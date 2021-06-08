# note that node = neuron, and activation functions = activation signal, for
    # comments here forth

# for plotting graphs
import matplotlib.pyplot as plt

# import package for libraries of basic NN algorithms with flexible network 
    # configurations and learning algorithms
import neurolab as nl

# input values
input = [[0, 0], [0, 1], [1, 0], [1, 1]] 

# target values of class 0 and 1
target = [[0], [0], [0], [1]]

# network with 2 inputs, and 1 neuron in output layer
net = nl.net.newp([[0, 1],[0, 1]], 1) 

# train the network using Delta rule
# input to network's input layer is input, output layer (labels) is target, 
    # epochs is num of iterations (later iterations will minimise errors & 
    # optimal output), show output every 10 iterations, 
    # learning rate = alpha = lr = 0.1 (higher means better outcome, help 
    # reach better outcome before later iterations)
error_progress = net.train(input, target, epochs=100, show=10, lr=0.1)
error_progress

# visualise output of training progress using error metric
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs') # x axis label
plt.ylabel('Training error') # y axis label
plt.grid()
plt.show()