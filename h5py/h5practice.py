"""
#importing required libraries 
import numpy as np
import h5py


#creating the matrix of size 1000,1000 by assigning random values 
matrix1=np.random.random(size=(1000,1000))
matrix2= np.random.random(size=(1000,1000))


#now let's create the hdff5 file!!!
with h5py.File('test.h5','w') as hdfwrite:
    dataset1=hdfwrite.create_dataset('dataset1',data=matrix1)
    dataset2=hdfwrite.create_dataset('dataset2',data=matrix2)


    #setting attributes to the datasets
    dataset1.attrs['CLASS']='DATA_MATRIX'
    dataset2.attrs['version']='1.1'
    hdfwrite.close()



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
print('hello')
"""
"""
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

data=hdfread.get('dataset')
dataset1=np.array(data)
#printing the shape of the dataset
print('shape of the dataset is ',dataset1.shape)	


    #read attributes 

k=list(data.attrs.keys())
v=list(data.attrs.values())
print(k[0])
print(v[0])



    #closig the connection of file 

"""
"""	
import numpy as np
import h5py

matrix1=np.random.random(size=(1000,1000))
matrix2=np.random.random(size=(1000,1000))
matrix3=np.random.random(size=(1000,1000))
matrix4=np.random.random(size=(1000,1000))

with h5py.File('hdf5_groups.h5','w') as hdf:
    G1=hdf.create_group('Group1')
    G1.create_dataset('datset1',data=matrix1)
    G1.create_dataset('dataset4',data=matrix4)
    
    G21=hdf.create_group('Group2/subgroup1')
    G21=hdf.create_dataset('dataset3',data=matrix3)
    
    G22=hdf.create_group('Group2/SubGroup2')
    G22=hdf.create_dataset('dataset2',data=matrix2)

    with h5py.File('hdf5_groups.h5','r') as hdf:
    base_items=list(hdf.items())
    print('Items in the dirextory :',base_items)

    """
    