

#%% Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    out = './Output/RF/'
    out1 = './Output/RF/RF/'
    f_path = './Output/BASE/'

    np.random.seed(0)
    ## Read in data ##
    nn_reg = [(10**-3)]
    nn_arch= [(14,)]
    nn_arch2= [(14,)]
    # nn_arch= [(12,),(14,),(16,)]
    # nn_arch2= [(4,),(6,),(8,)]

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
    dims = [2,3,4,5,6,7,8,9,10,12,15,20,25]
    
    #%% data for 1
    
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
    fs_faults = rfc.fit(faultsX,faultsY).feature_importances_ 
    fs_bc = rfc.fit(bcX,bcY).feature_importances_ 
    
    tmp = pd.Series(np.sort(fs_faults)[::-1])
    tmp.to_csv(out1+'faults scree.csv')
    
    tmp = pd.Series(np.sort(fs_bc)[::-1])
    tmp.to_csv(out1+'bc scree.csv')
    
    #%% Data for 2
    filtr = ImportanceSelect(rfc)
    grid ={'filter__n':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(faultsX,faultsY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out1+'faults dim red.csv')
    
    
    grid ={'filter__n':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch2}  
    mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(bcX,bcY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out1+'bc dim red.csv')
#    raise


    #%% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up


    dim = 24
    filtr = ImportanceSelect(rfc,dim)
    
    faultsX2 = filtr.fit_transform(faultsX,faultsY)
    faults2 = pd.DataFrame(np.hstack((faultsX2,np.atleast_2d(faultsY).T)))
    cols = list(range(faults2.shape[1]))
    cols[-1] = 'labels'
    faults2.columns = cols
    faults2.to_hdf(out+'datasets.hdf','train_faults',complib='blosc',complevel=9)
    
    dim = 10
    filtr = ImportanceSelect(rfc,dim)
    bcX2 = filtr.fit_transform(bcX,bcY)
    bc2 = pd.DataFrame(np.hstack((bcX2,np.atleast_2d(bcY).T)))
    cols = list(range(bc2.shape[1]))
    cols[-1] = 'diagnosis'
    bc2.columns = cols
    bc2.to_hdf(out+'datasets.hdf','train_bc',complib='blosc',complevel=9)