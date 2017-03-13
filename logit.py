#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:20:57 2017

@author: junewang
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import random as rd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
from sklearn.linear_model.logistic import  LogisticRegression



X = pd.read_csv('./train_dummy.csv')

y=pd.read_csv('./training.csv')['IsBadBuy']



from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=33)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#df = pd.read_csv('training.csv')
# y, X = dmatrices('IsBadBuy ~ PurchDate+Auction+VehYear+VehicleAge+Make+Model+Trim+SubModel+Color+Transmission+WheelTypeID+WheelType+VehOdo+Nationality+Size+TopThreeAmericanName+MMRAcquisitionAuctionAveragePrice+MMRAcquisitionAuctionCleanPrice+MMRAcquisitionRetailAveragePrice+MMRAcquisitonRetailCleanPrice+MMRCurrentAuctionAveragePrice+MMRCurrentAuctionCleanPrice+MMRCurrentRetailAveragePrice+MMRCurrentRetailCleanPrice+PRIMEUNIT+AUCGUART+BYRNO+VNZIP1+VNST+VehBCost+IsOnlineSale+WarrantyCost', df, return_type = 'dataframe')
model = LogisticRegression(fit_intercept = False, C = 1e9)

##cross_val_score
score = cross_val_score(model,X_train, y_train)
avg_acore = score.mean()

##test score
mdl = model.fit(X_train, y_train)
mdl.score(X_test,y_test)

##lasso
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train)
#Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
#   normalize=False, positive=False, precompute=False, random_state=None,
#   selection='cyclic', tol=0.0001, warm_start=False)
y_pred = clf.predict(X_test)


#print(clf.coef_)
#clf
#logit = sm.Logit(y, X)

#result = logit.fit()
#result.summary()


## make the amount of bad and good same

y_train_reset = y_train.reset_index(drop=True)
X_train_reset = X_train.reset_index(drop=True)
y_good_index = []
y_bad_index = []
index_new=[]
for i in range(0,len(y_train_reset)):
    if y_train_reset[i] == 0:
        y_good_index.append(i)
    else:
        y_bad_index.append(i) 

###(1)the numbers of good and bad are same
#rd.shuffle(y_good_index)
#y_good_index_new = rd.sample(y_good_index, len(y_bad_index))         
#index_new = y_bad_index + y_good_index_new
#index_new.extend(y_bad_index)
#rd.shuffle(index_new)

#y_train_new = y_train_reset[index_new]
#X_train_new = X_train_reset.loc[index_new,:]

###(2)choose the small number of each index and make a loop

index_new_all = []
for i in range(0,10):
    y_good_index_new = []
    y_bad_index_new = []
    index_new = []
    rd.shuffle(y_good_index)
    rd.shuffle(y_good_index)
    y_good_index_new = rd.sample(y_good_index, 2000)
    y_bad_index_new = rd.sample(y_bad_index, 2000)         
    index_new = y_good_index_new+y_bad_index_new
    rd.shuffle(index_new)
    index_new_all = index_new_all+index_new

y_train_new = y_train_reset[index_new_all]
X_train_new = X_train_reset.loc[index_new_all,:]

##logistical regression
coef_l1_LR_all = []
coef_l2_LR_all = []
sparsity_l1_LR_all = []
sparsity_l2_LR_all = []
score_l1_LR_all = []
score_l2_LR_all = []
C_all = np.linspace(0.001, 5, num=10)
for i, C in enumerate(C_all):
    clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
    clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
    clf_l1_LR.fit(X_train_new, y_train_new)
    clf_l2_LR.fit(X_train_new, y_train_new)
    
    coef_l1_LR = clf_l1_LR.coef_.ravel()
    coef_l2_LR = clf_l2_LR.coef_.ravel()
    coef_l1_LR_all.append(coef_l1_LR)
    coef_l2_LR_all.append(coef_l2_LR)
    
    sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
    sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100
    score_l1_LR = clf_l1_LR.score(X_test,y_test)
    score_l2_LR = clf_l2_LR.score(X_test,y_test)
    
    sparsity_l1_LR_all.append(sparsity_l1_LR)
    sparsity_l2_LR_all.append(sparsity_l2_LR)
    score_l1_LR_all.append(score_l1_LR)
    score_l2_LR_all.append(score_l2_LR)
    
    print("C=%.2f" % C)
    print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
    print("score with L1 penalty: %.4f" % score_l1_LR)
    print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
    print("score with L2 penalty: %.4f" % score_l2_LR)
plt.figure(1)
plt.subplot(211)
plt.plot(C_all[1:30], sparsity_l1_LR_all[1:30], 'k')
plt.plot(C_all[1:30], sparsity_l2_LR_all[1:30], 'r--')
plt.grid(True)
plt.figure(1)
plt.subplot(212)
plt.plot(C_all[1:30], score_l1_LR_all[1:30],'k')
plt.plot(C_all[1:30], score_l2_LR_all[1:30],'r--')
plt.grid(True)