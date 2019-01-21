# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:17:14 2017

@author: JTay
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#from helpers import nn_arch,nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = './Output/BASE/BASE/'
f_path = './Output/BASE/'

np.random.seed(0)

#Faults dataset
faults = pd.read_hdf(f_path+'datasets.hdf','train_faults')
faultsX = faults.drop('labels',1).copy().values
faultsY = faults['labels'].copy().values

#Breast Cancer dataset 
bc = pd.read_hdf(f_path+'datasets.hdf','train_bc')
bcX = bc.drop(['diagnosis'],1).copy().values
bcY = bc['diagnosis'].copy().values


faultsX = StandardScaler().fit_transform(faultsX)
bcX= StandardScaler().fit_transform(bcX)

nn_arch= [(14,)]
nn_reg = [(10**-3)]

#%% benchmarking for chart type 2

grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}

#Faults dataset
mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(faultsX,faultsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Faults NN bmk.csv')

#Breast Cancer dataset
mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5, learning_rate_init = 0.1, momentum = 0.3)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(bcX,bcY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'BreastC NN bmk.csv')
#raise