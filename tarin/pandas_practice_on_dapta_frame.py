import numpy as np
import pandas as pd



from sklearn.preprocessing  import LabelEncoder,OneHotEncoder


#define example 
data=['cold','cold','warm','cold','hot','hot','warm','cold','warm','hot']
values=np.array(data)
print(values)


#integer  encoding 

label_encoder=LabelEncoder()
integer_encoded=label_encoder.fit_transform(values)

#binary encoding
onehot_encoder=OneHotEncoder(sparse=True)
integer_encoder=
