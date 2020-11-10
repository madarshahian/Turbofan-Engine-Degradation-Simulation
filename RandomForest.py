# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:28:11 2020

@author: madar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
#%%
#%%
Train01 = pd.read_csv('CMAPSSData/train_FD001.txt', sep=" ", header=None)
Test01 = pd.read_csv('CMAPSSData/test_FD001.txt', sep=' ', header=None)
y_test = pd.read_csv('CMAPSSData/RUL_FD001.txt', sep=' ', header=None)
Train01=Train01.dropna(axis=1)
Test01=Test01.dropna(axis=1)
y_test=y_test.dropna(axis=1)
cols = ["Unit","Cycles"]
settings = [f'Set{i}' for i in range(1,4)]
sensors = [f'Sen{i}'.format(i) for i in range(1,22)]
Train01.columns = cols + settings + sensors
Test01.columns = cols + settings + sensors
y_test.columns= ['RUL']
Train01.head()
#%% Removing no info sensors
summary_sen = Train01[sensors].describe().T
active_sensors = list(summary_sen[summary_sen['max']!=summary_sen['min']].index)
print(f"active sensors are : {active_sensors}")
active_sensors.remove('Sen6')
#%%
max_units = Train01[['Unit','Cycles']].groupby(['Unit']).max().values[:,-1]
count_units = Train01[['Unit','Cycles']].groupby(['Unit']).count().values[:,-1]
a = np.empty(Train01.shape[0])
j=0
for item in zip(count_units,max_units):
    a[j:item[0]+j]=item[1]
    j+=item[0]
RUL = a - Train01['Cycles'].values
Train01['RUL'] = RUL
#%% Random Forest
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
#%%
X_train = Train01[active_sensors]
y_train = Train01[['RUL']]
scaler_X = preprocessing.StandardScaler().fit(X_train)
scaler_y = preprocessing.StandardScaler().fit(y_train)
X_train_scaled = scaler_X.transform(X_train)
y_train_scaled = scaler_y.transform(y_train)
#%%
clf = RandomForestRegressor()
clf.fit(X_train_scaled, y_train_scaled)
#%%
y_pre = clf.predict(X_train_scaled)
print(f"Coefficient of determination R^2 of the prediction for training data: {clf.score(X_train_scaled, y_train_scaled):2.2}")
X_test = Test01.groupby('Unit').last().reset_index()[active_sensors]
X_test_scaled = scaler_X.transform(X_test)
y_pre_test = clf.predict(X_test_scaled)
scaler_y_test = scaler_y.transform(y_test)
print(f"Coefficient of determination R^2 of the prediction for testing data: {clf.score(X_test_scaled, scaler_y_test):2.2}")
#%% tuning
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
#%%
clf.fit(X_train_scaled, y_train_scaled)
clf.best_params_
#%%
y_pre = clf.predict(X_train_scaled)
print(f"Coefficient of determination R^2 of the prediction for training data: {clf.score(X_train_scaled, y_train_scaled):2.2}")
X_test = Test01.groupby('Unit').last().reset_index()[active_sensors]
X_test_scaled = scaler_X.transform(X_test)
y_pre_test = clf.predict(X_test_scaled)
scaler_y_test = scaler_y.transform(y_test)
print(f"Coefficient of determination R^2 of the prediction for testing data: {clf.score(X_test_scaled, scaler_y_test):2.2}")

























