#1 Importing the libraries
import numpy as np  # import numpy for working with numbers
import pandas as pd
import matplotlib.pyplot as plt # import plots

#2 Importing the dataset:
dataset = pd.read_csv(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\Lab9_RegressionPracticalMaterialsV1.03\50_Startups.csv')

#Y: dependent variable vector
# X’s type is object due to the different types of independent variables. 
    # State column contains categorical variables
X= dataset.iloc[:, :-1].values
X
Y=dataset.iloc[:, 4].values
Y

# import encoder modules
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

# encoding the vategorical variable's text into numbers 0,1,2
X[: ,3]= labelencoder_X.fit_transform(X[: ,3])
X

# use one hot encoding
# onehotencoder= OneHotEncoder(categorical_features=[3])
onehotencoder = OneHotEncoder(categories='auto')

# convert the numbers to dummy variables. Each column represents one state, 
    # compare X and dataset tables to understand the relationship between 
    # the state and the columns
X= onehotencoder.fit_transform(X).toarray()
X

# Avoid the dummy variables trap
# Delete the first column representing California
X= X[:, 1:]
X
    
# import train test split module
from sklearn.model_selection import train_test_split

# train_set_split: Split arrays or matrices into random train and test subsets.
    # 20% test set, 80% train set
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=0)

# import linear regression model
from sklearn.linear_model import LinearRegression

# Create an object called regressor in the LinearRegression class
regressor = LinearRegression()

# Fit Multiple Linear Regression model to Train set
regressor.fit(X_train,Y_train)

# Predicting the Test set results:
y_pred = regressor.predict(X_test)
y_pred

# import statsmodels.formula.api as sm
import statsmodels.api as sm

# Beta0 has x0=1. Add a column of for the the first term of the MultiLinear 
    # Regression equation, contains only 1 in each 50 rows
X= np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1) 
X

# Optimal X contains the highly impacted independent variables
X_opt = X[:, [0,1,2,3,4,5]]

# OLS: Ordinary Least Square Class 
# endog is the dependent variable, exog is the number of observations
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# constant for Beta0, x1 and x2 are the dummy variables for state, x3 is R&D, 
    # x4 is Administration, x5 is the marketing spends 
    
# find and remove the highest p-value. remove x5 (fifth dummy variable) 
X_opt= X[:, [0,1,2,3,4]] 
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# find and remove the highest p-value. remove x4
X_opt= X[:, [0,1,2,3]] 
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# find and remove the highest p-value. remove x3
X_opt= X[:, [0,1,2]] 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# find and remove the highest p-value. remove x2
X_opt= X[:, [0,1]] 
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# find and remove the highest p-value. remove x1
X_opt= X[:, [0]] 
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()