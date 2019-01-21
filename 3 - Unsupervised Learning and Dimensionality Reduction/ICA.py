

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

out = './Output/ICA/'
out1 = './Output/ICA/ICA/'
f_path = './Output/BASE/'
np.random.seed(0)

nn_reg = [(10**-3)]
nn_arch= [(14,)]
nn_arch2= [(14,)]
# nn_arch= [(4,),(5,),(6,)]
# nn_arch2= [(14,),(15,),(16,)]
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
dims = [2,3,4,5,6,7,8,9,10,12,15,16,18,20,25]
#raise
#%% data for 1

ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(faultsX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out1+'faults scree.csv')


ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(bcX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out1+'bc scree.csv')
#raise

#%% Data for 2

grid ={'ica__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(faultsX,faultsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out1+'faults dim red.csv')


grid ={'ica__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch2}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(bcX,bcY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out1+'bc dim red.csv')
# raise


#%% data for 3
#Set this from chart 2 and dump, use clustering script to finish up
dim = 4
ica = FastICA(n_components=dim,random_state=10)

faultsX2 = ica.fit_transform(faultsX)
faults2 = pd.DataFrame(np.hstack((faultsX2,np.atleast_2d(faultsY).T)))
cols = list(range(faults2.shape[1]))
cols[-1] = 'labels'
faults2.columns = cols
faults2.to_hdf(out+'datasets.hdf','train_faults',complib='blosc',complevel=9)

dim = 25
ica = FastICA(n_components=dim,random_state=10)
bcX2 = ica.fit_transform(bcX)
bc2 = pd.DataFrame(np.hstack((bcX2,np.atleast_2d(bcY).T)))
cols = list(range(bc2.shape[1]))
cols[-1] = 'diagnosis'
bc2.columns = cols
bc2.to_hdf(out+'datasets.hdf','train_bc',complib='blosc',complevel=9)