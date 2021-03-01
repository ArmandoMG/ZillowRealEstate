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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#------------------------------------
#------------------------------------
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 20)

regressor = RandomForestRegressor(n_estimators = 300, min_samples_split=3, min_samples_leaf=1,max_features='sqrt',max_depth=80, bootstrap=True)
regressor.fit(X_train, y_train)

# {'n_estimators': 400,
#  'min_samples_split': 5,
#  'min_samples_leaf': 1,
#  'max_features': 'sqrt',
#  'max_depth': 100,
#  'bootstrap': False}

#RANDOMIZED SEARCH 
# {'n_estimators': 600,
#  'min_samples_split': 5,
#  'min_samples_leaf': 1,
#  'max_features': 'sqrt',
#  'max_depth': 60,
#  'bootstrap': False}
#Adj R2: 85.08
#Cross Val R2: 75.22 

#GRID SEARCH
# {'bootstrap': True,
#  'max_depth': 80,
#  'max_features': 'sqrt',
#  'min_samples_leaf': 1,
#  'min_samples_split': 3,
#  'n_estimators': 300}
#Adj R2: 84.00
#Cross Val R2: 74.75

#Improvement of 0.07%... not worth it


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid

rf = RandomForestRegressor()
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_
rf_random.best_score_ #77.96


param_grid = {
    'bootstrap': [True],
    'max_depth': [20, 80, 60, 90, 100, 110],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1,2],
    'min_samples_split': [3, 4, 5],
    'n_estimators': [20, 100, 200, 300, 400, 500, 700]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_ #77.78

#------------------------------------
#------------------------------------
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0,max_depth=12, min_samples_leaf=2, min_samples_split=10)
regressor.fit(X_train, y_train)
#GRID SEARCH
# {'bootstrap': True,
#  'max_depth': 12,
#  'min_samples_leaf': 2,
#  'min_samples_split': 10}
#Adj R2: 78.81
#Cross Val R2: 63.73

max_depth = [2,4,6,8,10,12]
param_grid = {
    'max_depth': max_depth,
    'min_samples_leaf': [1,2,4],
    'min_samples_split': [2, 4, 5, 10]
}

# Create a based model
dt = DecisionTreeRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_#68.41


#------------------------------------
#------------------------------------
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
regressor.fit(X_train, y_train)
#GRID SEARCH
#{'fit_intercept': True, 'normalize': True}
#Adj R2: 65.08
#Cross Val R2: 49.55

regressor.get_params()
param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}

mlr = LinearRegression()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = mlr, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_#54.27
#------------------------------------
#------------------------------------

# Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
#Adj R2: 67.17
#Cross Val R2: 56.66

#DO NOT RUN
# param_grid = {'C': [0.1, 1, 10, 100, 1000],  
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#               'kernel': ['poly', 'rbf', 'sigmoid']}

# regressor = SVR()
# grid_search = GridSearchCV(estimator = regressor, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)
# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
# grid_search.best_params_

#------------------------------------
#------------------------------------

# =============================================================================
# Visualizing Results
# =============================================================================

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred=np.exp(y_pred)
y_test=np.exp(y_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



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