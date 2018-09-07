#importing required libraries 
import numpy as np
import h5py


#creating the matrix of size 1000,1000 by assigning random values 
matrix1=np.random.random(size=(1000,1000))
maytrix2= np.random.random(size=(1000,1000))


#now let's create the hdff5 file!!!
with h5py.File('ml/test.h5','w') as hdfwrite:
dataset1=hdfwrite.create_dataset('dataset1',data=matrix1)
dataset2=hdfwrite.create_dataset('dataset2',data=matrix2)


#setting attributes to the datasets
dataset1.attrs['CLASS']='DATA_MATRIX'
dataset2.attrs['version']='1.1'
hdf.close()



#read hdf5 file 
with h5py.File('test.h5','r') as hdfread:
ls=list(hdfread.keys())
print('list of datasets in this files are\n',ls)

data=hdfread.get('dataset1')
dataset1=np.array(data)
#printing the shape of the dataset
print('shape of the dataset is ',dataset1.shape)	


    #read attributes 

k=list(data.attrs.keys())
v=list(data.attrs.values())
print(k[0])
print(v[0])



    #closig the connection of file 


