# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:17:44 2019

@author: pavan
"""

import numpy as np
import pandas as pd

#importing the dataset
dataset =pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values


#using the elbow method to find the optimal number of the clusters
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('no of Clusters')
plt.ylabel('wcss')
plt.legend()
plt.show()

#As we can conclude from the graph that we can best use the no of clusters as 3
#fit the model with the 3 no of clusters 

kmeans = KMeans(n_clusters=3,init='k-means++',n_init=10,random_state=0)
y_means = kmeans.fit_predict(X)


#visualizing the clusters of different type's
plt.scatter(X[y_means==0,0],X[y_means==0,1],c='red',s=100)
plt.scatter(X[y_means==1,0],X[y_means==1,1],c='green',s=100)
plt.scatter(X[y_means==2,0],X[y_means==2,1],c='blue',s=100)
plt.scatter(X[y_means==3,0],X[y_means==3,1],c='cyan',s=100)

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow' ,label='centroids')


