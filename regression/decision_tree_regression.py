# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 23:18:06 2019

@author: pavan
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')


#classifing data into dependent and independent
X = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X.reshape(-1,1),y.reshape(-1,1))

y_pred  =  regressor.predict(np.array([[4.5]]).reshape(-1,1))

#visualizing the decision tree  regression
import matplotlib.pyplot as plt

plt.scatter(X,y,color='r',marker='*')
plt.plot(X,regressor.predict(X.reshape(-1,1)),color='g')
plt.title("Decision Tree Regression")
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#visualizing the decision tree regression result  for higher resolution
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='g')
plt.plot(X_grid,regressor.predict(X_grid))
plt.title('Decision tree Regression')
plt.xlabel('Exeprience')
plt.ylabel('Salary')
plt.show()