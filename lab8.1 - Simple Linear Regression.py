# Importing the libraries
import numpy as np # for working with numbers
import matplotlib.pyplot as plt # for plotting graphs
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\Lab9_RegressionPracticalMaterialsV1.03\Salary_Data.csv')

# find attributes
X = dataset.iloc[:, :-1].values

# find target class label
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# 66.6% train set and 33.3% test set, using list of random number starting from position 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3.0, random_state = 0)

# import and use Linear Regression algorithm model from linear class
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor

# fit train set to Linear Regression model
regressor = regressor.fit(X_train, y_train)
regressor

# Predicting Test set results
y_pred = regressor.predict(X_test)
y_pred

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red') # scatter plot
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Linear Regression line
plt.title('Salary vs Experience (Training set)') # title of plot
plt.xlabel('Years of Experience') # x axis label
plt.ylabel('Salary') # y axis label
plt.show() # show plot

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red') # scatter plot
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Linear Regression line
plt.title('Salary vs Experience (Test set)') # title of plot
plt.xlabel('Years of Experience') # x axis label
plt.ylabel('Salary') # y axis label
plt.show() # show plot
