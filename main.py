# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 22:56:53 2021

@author: arman
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('modeling_properties.csv')

# =============================================================================
# Modeling
# =============================================================================
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y=np.log(y) #Como la distribucion no es normal, y la log distribucion si, usamos esa para un mejor fit

#Encoding categorical data
import category_encoders as ce
ct =  ce.BinaryEncoder(cols=0,return_df=False);#Encoding titles #calar despues con one hot tambien a ver qp
X= ct.fit_transform(X)

ct =  ce.BinaryEncoder(cols=4,return_df=False);#Encoding postal_codes #calar despues con one hot tambien a ver qp
X= ct.fit_transform(X)

#RFR: 84.71 - 74.64 or 75.01

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 20)
regressor.fit(X_train, y_train)
# {'n_estimators': 400,
#  'min_samples_split': 5,
#  'min_samples_leaf': 1,
#  'max_features': 'sqrt',
#  'max_depth': 100,
#  'bootstrap': False}

#Improvement of 0.07%... not worth it



# =============================================================================
# Visualizing Results
# =============================================================================

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred=np.exp(y_pred)
y_test=np.exp(y_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)
n=len(y_test)
p=len(X[0])
r2_adj=1-(1-r2)*(n-1)/(n-p-1)
print('R squared: ', r2)
print('Adjsuted R squared: ', r2_adj)

from sklearn.model_selection import cross_val_score
cross_val_score(regressor, X, y, cv=7, scoring='r2').mean()

#Predicting an arbitrary value
np.exp(regressor.predict([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 3.0, 2.0, 800]]))

plt.scatter(y_pred, y_test, color = 'blue')
plt.title('Predictions vs Test Results')
plt.xlabel('Predictions')
plt.ylabel('Test Results')
plt.show()




