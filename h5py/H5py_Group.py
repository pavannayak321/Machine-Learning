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

    