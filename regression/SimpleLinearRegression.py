# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:34:28 2019

@author: pavan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Salary_Data.csv')

#classifying the data into dependent And independent variables
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

#checking for null values 
dataset.isnull().sum()

#splitiing up the dataset into training and testing 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=1)#we usually set random set to see the performance of model in prtoduction split the training datset randomly  

#As we can see the salary is in the raw form we are not required too apply Feature Scaling
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#reshape(-1,1) indicates that row is unknown and column is exactly 1
regressor.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))


#predict the salary on expierience 
y_pred = regressor.predict(2.5)


