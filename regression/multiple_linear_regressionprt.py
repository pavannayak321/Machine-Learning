# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 21:43:10 2019

@author: pavan
"""
import numpy as np
import pandas as pd

#reading data from the  csv file 
dataset= pd.read_csv('C:/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')


#dividing the dataset into dependent and independent variables 
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,4].values

#checking for NAN values 
dataset.isnull().sum()
#applicable for whole dataset

#performing LabelEncoding  to the dataset
from sklearn.preprocessing import LabelEncoder
labelencoding = LabelEncoder()
X[:,3] = labelencoding.fit_transform(X[:,3])

#transforming the  data into OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
onehotencoding = OneHotEncoder(categorical_features=[3])
X = onehotencoding.fit_transform(X).toarray()

#performing the dummy variable trap
X = X[:,1:]

#splitting the dataset into 

#applying the feature scaling to the  training dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30)


#fitting the model with all possible features
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#trying to predict the resullts with  testing dataset ,compare with actual prediction
y_pred = regressor.predict(X_test)

#Building the optimal model (Performance reason,extract significant models)
import statsmodels.formula.api as sm
X = X[:,1:-1]
'''
appending the dependent variables with ones for coefficient of x0
'''
X = np.append(arr = np.ones((50,1)).astype(int ),values=X , axis = 1)
X_optimal_features =X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_optimal_features).fit()
regressor_OLS.summary()

'''
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
==============================================================================
'''
#since we can remove the variable that has highest p value>0.05---X2
X_optimal_features =X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_optimal_features).fit()
regressor_OLS.summary()

'''
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138
x2             0.8060      0.046     17.606      0.000       0.714       0.898
x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
x4             0.0270      0.017      1.592      0.118      -0.007       0.061
==============================================================================
'''
X_optimal_features =X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_optimal_features).fit()
regressor_OLS.summary()

'''
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
x1             0.8057      0.045     17.846      0.000       0.715       0.897
x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076
x3             0.0272      0.016      1.655      0.105      -0.006       0.060
==============================================================================
'''
X_optimal_features =X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_optimal_features).fit()
regressor_OLS.summary()
'''
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061
==============================================================================
'''

