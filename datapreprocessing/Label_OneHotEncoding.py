# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script for basic imputation of missing data in a frame  file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#reading data from csv file
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values
 



#taking the care of the miossing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])


#encoding categorical variables 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#lalbel encoding of Y
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)