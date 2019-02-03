# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 12:09:39 2019

@author: pavan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values


#using the dendograms  to find the nnumber of optimal clusters
import scipy.cluster.hierarchy as sch
dendogram =sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customer')
plt.ylabel('Eucliden Distnce')
plt.show()

#fitting the hierarcical clustering with clusters ==5
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean' ,linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualizing the clustering

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],c='red',s=100)
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],c='blue',s=100)
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],c='green',s=100)
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],c='cyan',s=100)
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],c='magenta',s=100)


plt.title('Cluster of Clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()