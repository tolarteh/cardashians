# Author: 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
from sklearn.linear_model.logistic import  LogisticRegression


data = pd.read_csv('./total.csv')
for feature in [ 'RefId','PurchDate','VehYear','Model', 'Trim', 'SubModel', 'WheelType', 'VNZIP1']:
    data.drop([feature],axis=1,inplace=True)


feature_names = data.columns[0:].tolist()
print('data_shape', data.shape)
print('there are {} items'.format(data.shape[0]))
print('there are {} charicteristics:{}'.format(len(feature_names),feature_names))

## check the features which contain missing value

missing_No = {}
for feature in feature_names:
    if len(data[feature][data[feature].isnull()]) > 0:
        print(feature)
        print(len(data[feature][data[feature].isnull()]))
        missing_No[feature] = len(data[feature][data[feature].isnull()])
print(missing_No)

## filling the missing value for catrgory features.

for feature in ['Color', 'Transmission', 'WheelTypeID', 'Nationality', 'Size', 'TopThreeAmericanName','PRIMEUNIT', 'AUCGUART']:
    data[feature][data[feature].isnull()]='U0'

## filling the missing value with average value for number features
for key in missing_No:
    if data[key].dtype !='object':
       data[key][data[key].isnull()]=data[key].median()

data = pd.get_dummies(data)
#print(data.head())

#separate training and test
train = data.loc[0:72982]
test = data.loc[72983:]


#print(test)
train.to_csv('./train_dummy.csv',index=False,header=True)
test.to_csv('./test_dummy.csv',index=False,header=True)
