# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:59:29 2018

@author: Pavan Nayak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#loading data into the pyhton prograamme 
dataset=pd.read_csv('50_Startups.csv')

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#label encoding to the training dataset
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder=LabelEncoder()
X[:,3]=label_encoder.fit_transform(X[:,3])
onehot_encoder=OneHotEncoder(categorical_features=[3])
X=onehot_encoder.fit_transform(X).toarray()

#avoiding the dummy variable traping
X=X[:,1:]

#spliting the dataset into training and spliting dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#fitting the linear regresion model 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#building the optimal model using the Backward Eliminaation method
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)


X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
