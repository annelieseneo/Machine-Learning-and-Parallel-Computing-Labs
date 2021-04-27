#  numpy is used as a general purpose array-processing package 
    # to efficiently manipulate large multi-dimensional arrays
import numpy as np 

# import preprocessing for common utility functions and transformer classes
from sklearn import preprocessing 

# create an array as sample data, and assign to input_data
input_data = np.array([[2.1, -1.9, 5.5], 
                       [-1.5, 2.4, 3.5], 
                       [0.5, -7.9, 5.6], 
                       [5.9, 2.3, -5.8]]) 

# binarizing data of the array input_data
# using the preprocessing library, set threshold to 0.5, where values 
    # greater than 0.5 will undergo binarisation transformation to 1, 
    # and those equal or less than 0.5 to 0
data_binarized = preprocessing.Binarizer(threshold=0.5).transform(input_data)

# print binarized data
print("\nBinarized data:\n", data_binarized)

# print original array input_data for comparison
print(input_data)

# 0 indicates column. print the mean of each column
print("Mean =", input_data.mean(axis=0))

# print the standard deviation of each column
print("Std deviation = ", input_data.std(axis=0))

# Mean Removal to eliminate the mean from feature vector so that 
    # every feature is centered on zero, and to remove bias from the features 
data_scaled = preprocessing.scale(input_data) 
print("Mean =", data_scaled.mean(axis=0)) 

# standard deviation removal
print("Std deviation =", data_scaled.std(axis=0))

# Scaling of feature vectors to prevent synthetically large or small features
# define min-max scaling, range is between 0 and 1
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))

# min-max scaling
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data) 

# print min-max scaled data
print ("\nMin max scaled data:\n", data_scaled_minmax) 

# normalisation to measure feature vectors on a common scale
# L1 Normalization or Least Absolute Deviations. modifies the values such that 
    # the sum of absolute values will be between -1 and 1 for each row
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')

# print L1 Normalized data
print("\nL1 normalized data:\n", data_normalized_l1) 

# L2 Normalization or least squares. modifies the values such that 
    # the sum of squares will be between -1 and 1 for each row
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')

# print L2 Normalized data
print("\nL2 normalized data:\n", data_normalized_l2) 



# define sample input labels to be used to create and train the label encoder
input_labels = ['red','black','red','green','black','yellow','white']

# create the label encoder object
encoder = preprocessing.LabelEncoder()

# train the label encoder object
encoder.fit(input_labels)

# create a random ordered list of labels
test_labels = ['green','red','black'] 

# check performance by encoding random ordered list of labels
encoded_values = encoder.transform(test_labels) 

# print the random ordered list of labels
print("\nLabels =", test_labels) 

# print the list of encoded values
print("Encoded values =", list(encoded_values))

# creating a random set of numbers
encoded_values = [3,0,4,1] 

# decoding a random set of numbers
decoded_list = encoder.inverse_transform(encoded_values) 

# print encoded values
print("\nEncoded values =", encoded_values)

# print decoded values
print("\nDecoded labels =", list(decoded_list))