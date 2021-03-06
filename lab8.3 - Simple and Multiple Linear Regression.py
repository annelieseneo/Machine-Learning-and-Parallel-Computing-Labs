import pandas as pd

# read the csv file
df = pd.read_csv(r'C:\Users\annel\Downloads\ITS66604 Machine Learning and Parallel Computing\labs\Lab9_RegressionPracticalMaterialsV1.03\weight-height.csv')

# first 5 rows of the data set
df.head()

# shape of the dataframe
df.shape

# data type of each column
df.dtypes

# number of null values
df.info()

# number of unique values of column Gender
df.Gender.nunique()
# 2

# unique values of column Gender
df.Gender.unique()
# array(['Male', 'Female'], dtype=object)

# import plots
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# histogram of Height
df.Height.plot(kind='hist', color='purple', edgecolor='black', figsize=(10,7))
plt.title('Distribution of Height', size=24) # title of plot
plt.xlabel('Height (inches)', size=18) # x axis label
plt.ylabel('Frequency', size=18) # y axis label

# histogram of Weight
df.Weight.plot(kind='hist', color='purple', edgecolor='black', figsize=(10,7))
plt.title('Distribution of Weight', size=24) # title of plot
plt.xlabel('Weight (pounds)', size=18) # x axis label
plt.ylabel('Frequency', size=18) # y axis label

# histogram of Height - males and females
df[df['Gender'] == 'Male'].Height.plot(kind='hist', color='blue', edgecolor='black', 
                                       alpha=0.5, figsize=(10, 7))
df[df['Gender'] == 'Female'].Height.plot(kind='hist', color='magenta', 
                                         edgecolor='black', alpha=0.5, 
                                         figsize=(10, 7))
plt.legend(labels=['Males', 'Females']) # legend
plt.title('Distribution of Height', size=24) # title of plot
plt.xlabel('Height (inches)', size=18) # x axis label
plt.ylabel('Frequency', size=18) # y axis label

# histogram of Weight - males and females
df[df['Gender'] == 'Male'].Weight.plot(kind='hist', color='blue', edgecolor='black', 
                                       alpha=0.5, figsize=(10, 7))
df[df['Gender'] == 'Female'].Weight.plot(kind='hist', color='magenta', 
                                         edgecolor='black', alpha=0.5, 
                                         figsize=(10, 7))
plt.legend(labels=['Males', 'Females']) # legend
plt.title('Distribution of Weight', size=24) # title of plot
plt.xlabel('Weight (pounds)', size=18) # x axis label
plt.ylabel('Frequency', size=18) # y axis label

# Descriptive statistics for male
statistics_male = df[df['Gender'] == 'Male'].describe()
statistics_male.rename(columns=lambda x: x + '_male', inplace=True)

# Descriptive statistics for female
statistics_female = df[df['Gender'] == 'Female'].describe()
statistics_female.rename(columns=lambda x: x + '_female', inplace=True)

# Dataframe that contains statistics for both male and female
statistics = pd.concat([statistics_male, statistics_female], axis=1)
statistics

# Scatter plot of Height and Weight
ax1 = df[df['Gender'] == 'Male'].plot(kind='scatter', x='Height', y='Weight', 
                                      color='blue', alpha=0.5, figsize=(10, 7))
df[df['Gender'] == 'Female'].plot(kind='scatter', x='Height', y='Weight', 
                                  color='magenta', alpha=0.5, figsize=(10 ,7), 
                                  ax=ax1)
plt.legend(labels=['Males', 'Females']) # legend
plt.title('Relationship between Height and Weight', size=24) # title of plot
plt.xlabel('Height (inches)', size=18) # x axis label
plt.ylabel('Weight (pounds)', size=18); # y axis label

# Scatter plot of 500 females
sample_females = df[df['Gender'] == 'Female'].sample(500) # 500 data points
sample_females.plot(kind='scatter', x='Height', y='Weight', color='magenta', 
                    alpha=0.5, figsize=(10, 7))
plt.legend(labels=['Females']) # legend
plt.title('Relationship between Height and Weight (sample of 500 females)', 
          size=20) # title of plot
plt.xlabel('Height (inches)', size=18) # x axis label
plt.ylabel('Weight (pounds)', size=18); # y axis label

# import numpy for working with numbers
import numpy as np

# best fit polynomials
df_males = df[df['Gender'] == 'Male']
df_females = df[df['Gender'] == 'Female']

# polynomial - males
male_fit = np.polyfit(df_males.Height, df_males.Weight, 1)
male_fit
# array([ 5.96177381, -224.49884071])

# polynomial - females
female_fit = np.polyfit(df_females.Height, df_females.Weight, 1)
female_fit
# array([ 5.99404661, -246.01326575])

# scatter plots and regression lines
# males and females dataframes
df_males = df[df['Gender'] == 'Male']
df_females = df[df['Gender'] == 'Female']

# Scatter plots
ax1 = df_males.plot(kind='scatter', x='Height', y='Weight', color='blue', 
                    alpha=0.5, figsize=(10, 7))
df_females.plot(kind='scatter', x='Height', y='Weight', color='magenta', 
                alpha=0.5, figsize=(10, 7), ax=ax1)

# regression lines
plt.plot(df_males.Height, male_fit[0] * df_males.Height + male_fit[1], 
         color='darkblue', linewidth=2)
plt.plot(df_females.Height, female_fit[0] * df_females.Height + female_fit[1], 
         color='deeppink', linewidth=2)

# regression equations
plt.text(65, 230, 'y={:.2f}+{:.2f}*x'.format(male_fit[1], male_fit[0]), 
         color='darkblue', size=12)
plt.text(70, 130, 'y={:.2f}+{:.2f}*x'.format(female_fit[1], female_fit[0]), 
         color='deeppink', size=12)

# legend, title and labels
plt.legend(labels=['Males Regresion Line', 'Females Regresion Line', 
                   'Males', 'Females']) # legend
plt.title('Relationship between Height and Weight', size=24) # title of plots
plt.xlabel('Height (inches)', size=18) # x axis label
plt.ylabel('Weight (pounds)', size=18) # y axis label

# import searborn library for more variety of data visualisation using 
# fewer syntax and interesting default themes
import seaborn as sns

# regression plot using seaborn
fig = plt.figure(figsize=(10, 7)) # fig size
sns.regplot(x=df_males.Height, y=df_males.Weight, color='blue', marker='+')
sns.regplot(x=df_females.Height, y=df_females.Weight, color='magenta', marker='+')

# Legend, title and labels.
plt.legend(labels=['Males', 'Females']) # legend
plt.title('Relationship between Height and Weight', size=24) # title of plot
plt.xlabel('Height (inches)', size=18) # x axis label
plt.ylabel('Weight (pounds)', size=18) # y axis label

# 300 random samples for Male and Female
df_males_sample = df[df['Gender'] == 'Male'].sample(300)
df_females_sample = df[df['Gender'] == 'Female'].sample(300)

# regression plot using seabornfig = plt.figure(figsize=(10, 7))
sns.regplot(x=df_males_sample.Height, y=df_males_sample.Weight, color='blue', 
            marker='+')
sns.regplot(x=df_females_sample.Height, y=df_females_sample.Weight, 
            color='magenta', marker='+')
plt.legend(labels=['Males', 'Females']) # legend
plt.title('Relationship between Height and Weight', size=24) # title of plot
plt.xlabel('Height (inches)', size=18) # x axis label
plt.ylabel('Weight (pounds)', size=18) # y axis label

# import linear regression model
from sklearn.linear_model import LinearRegression

# dataframe of Males
df_males = df[df['Gender'] == 'Male']

# create linear regression object
lr_males = LinearRegression()

# fit linear regression
lr_males.fit(df_males[['Height']], df_males['Weight'])

# intercept of the best fit line
print(lr_males.intercept_)
# -224.49884070545772

# slope of the best fit line
print(lr_males.coef_)
# 5.96177381

# dataframe of Female
df_females = df[df['Gender'] == 'Female']

# create linear regression object
lr_females = LinearRegression()

# fit linear regression
lr_females.fit(df_females[['Height']], df_females['Weight'])

# intercept of the best fit line
print(lr_females.intercept_)
# -246.01326574667277

# slope of the best fit line
print(lr_females.coef_)
# 5.99404661

# dataframe of Female
df_females = df[df['Gender'] == 'Female']

# fit the model using numpy
female_fit = np.polyfit(df_females.Height, df_females.Weight, 1)
female_fit

# predictions using numpy
print(np.polyval(female_fit, [60]))
# [113.62953114]

# fit the model using scikit learnlr_females = LinearRegression()
lr_females.fit(df_females[['Height']], df_females['Weight'])

# predictions using scikit learn
print(lr_females.predict([[60]]))
# [113.62953114]

# dataframe of Female
df_females = df[df['Gender'] == 'Female']

# correlation coefficients 
df_females.corr()

# dataframe of Male
df_males = df[df['Gender'] == 'Male']

# correlation coefficients 
df_males.corr()

from scipy import stats

# dataframe of Female
df_females = df[df['Gender'] == 'Female']

# pearson correlation coefficient and p-value
pearson_coef, p_value = stats.pearsonr(df_females.Height, df_females.Weight)
print(pearson_coef)
# 0.849608591418601

# dataframe of Male
df_males = df[df['Gender'] == 'Male']

# pearson correlation coefficient and p-value
pearson_coef, p_value = stats.pearsonr(df_males.Height, df_males.Weight)
print(pearson_coef)
# 0.8629788486163176

# dataframe of 500 data points of Female
df_females = df[df['Gender'] == 'Female'].sample(500)

# residual plot 500 females
fig = plt.figure(figsize = (10, 7)) # fig size
sns.residplot(df_females.Height, df_females.Weight, color='magenta')
plt.title('Residual plot 500 females', size=24) # title of plot
plt.xlabel('Height (inches)', size=18) # x axis label
plt.ylabel('Weight (pounds)', size=18) # y axis label

# dataframe of 500 data points of Males
df_males = df[df['Gender'] == 'Male'].sample(500)

# residual plot of 500 males
fig = plt.figure(figsize=(10, 7)) # fig size
sns.residplot(df_males.Height, df_males.Weight, color='blue')
plt.title('Residual plot 500 males', size=24) # title of plot
plt.xlabel('Height (inches)', size=18) # x axis label
plt.ylabel('Weight (pounds)', size=18) # y axis label

# print(df)
df_dummy = pd.get_dummies(df)
print(df_dummy)
# df_dummy = pd.get_dummies(df, dummy_na=True)

# drop female column
df_dummy.drop('Gender_Female', axis=1, inplace=True)

# rename Gender_Male column
df_dummy.rename(columns={'Gender_Male': 'Gender'}, inplace=True)

# df_dummy dataframe first 5 columns
df_dummy.head()

# create linear regression object
mlr = LinearRegression()

# fit linear regression
mlr.fit(df_dummy[['Height', 'Gender']], df_dummy['Weight'])

# intercept of the best fit line
print(mlr.intercept_)
# -244.92350252069903

# slope of the best fit line
print(mlr.coef_)
# [ 5.97694123 19.37771052