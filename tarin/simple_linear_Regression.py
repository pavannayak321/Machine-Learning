# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 18:38:15 2018

@author: Pavan Nayak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values


#spliting the data into train and test
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


#fitting the simple linearRegression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))


#predicting the test result
y_predict=regressor.predict(X_test.reshape(-1,1))
regressor.score(X_test.reshape(-1,1),y_test.reshape(-1,1))

#visualizing the training set model fitting 
plt.scatter(X_train.reshape(-1,1),y_train.reshape(-1,1),color='red')
plt.plot(X_train.reshape(-1,1),regressor.predict(X_train.reshape(-1,1)))
plt.title("Salary Prediction")
plt.xlabel("Year Of Expe")
plt.ylabel("Salary")
plt.show()





#visualizing the test set model fitting 
plt.scatter(X_train.reshape(-1,1),y_train.reshape(-1,1),color='red')
plt.plot(X_train.reshape(-1,1),regressor.predict(X_train.reshape(-1,1)))
plt.title(" Test set Salary Prediction")
plt.xlabel("Year Of Expe")
plt.ylabel("Salary")
plts.show()
