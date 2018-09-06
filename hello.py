import pandas as pd


#defining the mid-1 marks of the student

#framing the data of mid-1
mid1=pd.DataFrame({
                    "names":['pavan','kareem','ajay','anil','abhi'],
                    "roll_num":[1249,1201,1206,1210,1203],
                    "marks":[22,22,32,35,34]
                    },index=[201,202,203,204,205])

print(mid1)
print('---------------------------------------------')
#definingthe second mid marks

#framing the second mid-2 marks

mid2=pd.DataFrame({
                    "names1":['pavan','kareem','ajay','anil','abhi'],
                    "roll_num1":[1249,1201,1206,1210,1203],
                    "helo1":[1,2,3,4,5],
                    "marks1":[21,34,43,23,23]},index=[211,222,213,224,205])
print("printing the alues without change in the coumn type ")
print("-----------------------------------------------------")
#print(mid2)

#renaminng the column name in dataframe


mid1 =mid1.rename(columns={"names":"aliens"})
print(mid1)
concat=pd.concat([mid1,mid2])
print(concat)


'''
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


#taking the care of the miossing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

'''