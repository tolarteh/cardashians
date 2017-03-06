import pandas as pd
import numpy as np
X = pd.read_csv('./data_preprocessing/train_dummy.csv')

y=pd.read_csv('./data_preprocessing/training.csv')['IsBadBuy']

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train,y_train)
prediction_dt = dt.predict(X_test)
accurancy = dt.score(X_test, y_test)
print('the score with all features is %f' %accurancy)

#using feature selection
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)

X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
accurancy = dt.score(X_test_fs, y_test)
print('the score with 20 percent features is %f' %accurancy)

##find the selection rate for highest accurancy
from sklearn.cross_validation import cross_val_score
percentiles = range(1, 100, 5)

results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile = i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())
print(results)

opt = np.where(results == results.max())[0]
print('Optimal percentage of features %d' %percentiles[opt])
import pylab as pl

# pl.plot(percentiles, results)
# pl.show()

#using the rate that obatin the highest accurancy
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=97)

X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
highest_accurancy = dt.score(X_test_fs, y_test)
print(highest_accurancy)