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
