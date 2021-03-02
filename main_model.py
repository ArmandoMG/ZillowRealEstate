# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:04:57 2021

@author: arman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('modeling_properties.csv')

dataset.info()
# =============================================================================
# Modeling
# =============================================================================
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y=np.log(y) #Since the distribution its not normal, and log distribution it is, we'll use log for a better fit, we'll have to remember to transform this for the predictions


#Encoding categorical data
import category_encoders as ce
ct =  ce.BinaryEncoder(cols=0,return_df=False);#Encoding titles #calar despues con one hot tambien a ver qp
X= ct.fit_transform(X)

ct =  ce.BinaryEncoder(cols=4,return_df=False);#Encoding postal_codes #calar despues con one hot tambien a ver qp
X= ct.fit_transform(X)

#RFR: 85.19 - 75.00-75.37

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
#This regressor is not much of a improvement to the original, 1-2%. But i'll use it for the sake of the project and explanation. Normally I'd use the other model since its almost the same
regressor = RandomForestRegressor(n_estimators = 300, min_samples_split=5, min_samples_leaf=1,max_features='sqrt',max_depth=60, bootstrap=False)
regressor.fit(X_train, y_train)

# =============================================================================
# Visualizing Results
# =============================================================================

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred_normal=np.exp(y_pred)
y_test_normal=np.exp(y_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print(np.concatenate((y_pred_normal.reshape(len(y_pred_normal),1), y_test_normal.reshape(len(y_test_normal),1)),1))


# Evaluating the Model Performance
from sklearn.metrics import mean_absolute_error, r2_score
r2=r2_score(y_test, y_pred)
n=len(y_test)
p=len(X[0])
r2_adj=1-(1-r2)*(n-1)/(n-p-1)
print('R squared: ', r2)
print('Adjsuted R squared: ', r2_adj)
print('MAE: ', mean_absolute_error(np.exp(y_test), np.exp(y_pred)))

from sklearn.model_selection import cross_val_score
cross_val_score(regressor, X, y, cv=7, scoring='r2').mean()
cross_val_score(regressor, X, y, cv=7, scoring='explained_variance').mean()
cross_val_score(regressor, X, y, cv=7, scoring='neg_mean_absolute_error').mean()

#Predicting an arbitrary value
result = np.exp(regressor.predict([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 3.0, 2.0, 800]]))
print('Arbitrary Prediction (0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 3.0, 2.0, 800): ', result)

plt.scatter(y_pred, y_test, color = 'blue')
plt.title('Predictions vs Test Results')
plt.xlabel('Predictions')
plt.ylabel('Test Results')
plt.show()