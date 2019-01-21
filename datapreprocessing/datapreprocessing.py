# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy="mean",axis=0)
imputer.fit(X[:,[1,2]])
X[:,[1,2]] = imputer.transform(X[:,[1,2]])

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0])

