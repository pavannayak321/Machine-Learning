# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

#figuring out the  orientation of the data
import matplotlib.pyplot as  plt
plt.scatter(X,y,color="red")
plt.show()

#the model which best fits the data is SVR with kernel ,polynomial,sigmoid
from sklearn.svm import SVR
regressor = SVR(kernel = 'poly')
regressor.fit(X.reshape(-1,1),y.reshape(-1,1))


#predicting the outcome with experience 6.5yrs

y_pred = regressor.predict(np.array([[6.5]]))

#visualize the graph
plt.scatter(X,y)
plt.plot(X,regressor.predict(X.reshape(-1,1)),color="green")
plt.show()