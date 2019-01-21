# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:51:37 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#from helpers import  nn_arch,nn_reg
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

out = './Output/PCA/'
out1 = './Output/PCA/PCA/'
f_path = './Output/BASE/'
cmap = cm.get_cmap('Spectral') 

np.random.seed(0)

#Set from chart:
nn_reg = [(10**-3)]
# nn_arch= [(12,),(14,)]
# nn_arch2= [(3,),(4,),(6,)]
nn_arch= [(14,)]
nn_arch2= [(14,)]
## Read in data ##

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

clusters =  [2,3,4,5,6,7,8,9,10]
dims = [2,3,4,5,6,7,8,9,10,12,15]
#raise
#%% data for 1

## PCA for Faults dataset
pca = PCA(random_state=5)
pca.fit(faultsX)

#Creates a series with data = pca explained variance in the index = range
tmp = pd.Series(data = pca.explained_variance_,index = range(1,27))
tmp.to_csv(out1+'faults scree.csv')


## PCA for Breast Cancer dataset
pca = PCA(random_state=5)
pca.fit(bcX)

#Creates a series with pca explained variance in the range
tmp = pd.Series(data = pca.explained_variance_,index = range(1,31))
tmp.to_csv(out1+'bc scree.csv')



#%% Data for 2

###### Part 4 - Apply dim red on one of your datasets 


grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(faultsX,faultsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out1+'faults dim red.csv')


grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch2}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(bcX,bcY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out1+'bc dim red.csv')
#raise



#%% data for 3
#Set this from chart 2 and dump, use clustering script to finish up


dim = 12
pca = PCA(n_components=dim,random_state=10)

faultsX2 = pca.fit_transform(faultsX)
faults2 = pd.DataFrame(np.hstack((faultsX2,np.atleast_2d(faultsY).T)))
cols = list(range(faults2.shape[1]))
cols[-1] = 'labels'
faults2.columns = cols
faults2.to_hdf(out+'datasets.hdf','train_faults',complib='blosc',complevel=9)

dim = 7
pca = PCA(n_components=dim,random_state=10)
bcX2 = pca.fit_transform(bcX)
bc2 = pd.DataFrame(np.hstack((bcX2,np.atleast_2d(bcY).T)))
cols = list(range(bc2.shape[1]))
cols[-1] = 'diagnosis'
bc2.columns = cols
bc2.to_hdf(out+'datasets.hdf','train_bc',complib='blosc',complevel=9)