# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:38:28 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import silhouette_score as ss
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys


#sys.argv[1] represents the first command-line argument (as a string) supplied to the script in question.
out = './Output/{}/'.format(sys.argv[1])



#L2 Penalty Parameter Regularization for NN
nn_reg = [(10**-3)]


#Set hidden layer neurons based on dimensions of results
arg = sys.argv[1]

#nn_arch = fautly plates dataset
#nn_arch2 = breast cancer dataset

if (arg=="BASE"):
     nn_arch= [(14,)]
     nn_arch2= [(14,)]
# if (arg=="PCA"):
#     nn_arch= [(6,),(8,),(4,)]
#     nn_arch2= [(3,),(4,),(6,)]
# if (arg=="ICA"):
#     nn_arch= [(4,),(5,),(6,)]
#     nn_arch2= [(14,),(15,),(16,)]
# if (arg=="RP"):
#     nn_arch= [(6,),(8,),(4,)]
#     nn_arch2= [(3,),(4,),(6,)]
# if (arg=="RF"):
#     nn_arch= [(12,),(14,),(16,)]
#     nn_arch2= [(4,),(6,),(8,)]


np.random.seed(0)

## Read in data ##

#Faults dataset
faults = pd.read_hdf(out+'datasets.hdf','train_faults')        
faultsX = faults.drop('labels',1).copy().values
faultsY = faults['labels'].copy().values

#bc dataset
bc = pd.read_hdf(out+'datasets.hdf','train_bc')     
bcX = bc.drop(['diagnosis'],1).copy().values
bcY = bc['diagnosis'].copy().values

faultsX = StandardScaler().fit_transform(faultsX)
bcX= StandardScaler().fit_transform(bcX)

#clusters =  [2,5,8,10,15,20,25,30,40,50]

clusters =  [2,3,4,5,6,7,8,9,10]

#%% Data for 1-3
SSE = defaultdict(dict)
ll = defaultdict(dict)
SS = defaultdict(lambda: defaultdict(dict))
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(faultsX)
    gmm.fit(faultsX)


    #Faults dataset
    #Visual Measurements
    #Sum of Squared Errors for K-means
    SSE[k]['Faults'] = km.score(faultsX)

    #Log-Likelihood for GMM
    ll[k]['Faults'] = gmm.score(faultsX)

    #Silhouette Score
    #The best value is 1 and the worst value is -1. Silhouette analysis can be used to study the separation distance between the resulting clusters.
    SS[k]['Faults']['Kmeans'] = ss(faultsX, km.predict(faultsX))
    SS[k]['Faults']['GMM'] = ss(faultsX, gmm.predict(faultsX))
    #Cluster Accuracy    
    acc[k]['Faults']['Kmeans'] = cluster_acc(faultsY,km.predict(faultsX))
    acc[k]['Faults']['GMM'] = cluster_acc(faultsY,gmm.predict(faultsX))

    #Adjusted Mutual Information
    adjMI[k]['Faults']['Kmeans'] = ami(faultsY,km.predict(faultsX))
    adjMI[k]['Faults']['GMM'] = ami(faultsY,gmm.predict(faultsX))

    #Breast Cancer dataset
    km.fit(bcX)
    gmm.fit(bcX)
    SSE[k]['BreastC'] = km.score(bcX)
    ll[k]['BreastC'] = gmm.score(bcX)
    SS[k]['BreastC']['Kmeans'] = ss(bcX, km.predict(bcX))
    SS[k]['BreastC']['GMM'] = ss(bcX, gmm.predict(bcX))
    acc[k]['BreastC']['Kmeans'] = cluster_acc(bcY,km.predict(bcX))
    acc[k]['BreastC']['GMM'] = cluster_acc(bcY,gmm.predict(bcX))
    adjMI[k]['BreastC']['Kmeans'] = ami(bcY,km.predict(bcX))
    adjMI[k]['BreastC']['GMM'] = ami(bcY,gmm.predict(bcX))
    print(k, clock()-st)
    
    
SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
SS = pd.Panel(SS)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)


SSE.to_csv(out+'SSE.csv')
ll.to_csv(out+'logliklihood.csv')
SS.ix[:,:,'Faults'].to_csv(out+'Faults Silhouette.csv')
SS.ix[:,:,'BreastC'].to_csv(out+'BreastC Silhouette.csv')
acc.ix[:,:,'Faults'].to_csv(out+'Faults acc.csv')
acc.ix[:,:,'BreastC'].to_csv(out+'BreastC acc.csv')
adjMI.ix[:,:,'Faults'].to_csv(out+'Faults adjMI.csv')
adjMI.ix[:,:,'BreastC'].to_csv(out+'BreastC adjMI.csv')



# #%% NN fit data (2,3)
if (arg=="BASE"):
    grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
    km = kmeans(random_state=5)
    pipe = Pipeline([('km',km),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10)

    gs.fit(faultsX,faultsY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'Faults cluster Kmeans.csv')


    grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
    gmm = myGMM(random_state=5)
    pipe = Pipeline([('gmm',gmm),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(faultsX,faultsY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'Faults cluster GMM.csv')


    grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch2}
    mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
    km = kmeans(random_state=5)
    pipe = Pipeline([('km',km),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(bcX,bcY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'BreastC cluster Kmeans.csv')


    grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch2}
    mlp = MLPClassifier(activation='relu',max_iter=200,early_stopping=True,random_state=5,learning_rate_init = 0.1, momentum = 0.3)
    gmm = myGMM(random_state=5)
    pipe = Pipeline([('gmm',gmm),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(bcX,bcY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'BreastC cluster GMM.csv')


# # %% For chart 4/5
faultsX2D = TSNE(verbose=10,random_state=5).fit_transform(faultsX)
bcX2D = TSNE(verbose=10,random_state=5).fit_transform(bcX)

faults2D = pd.DataFrame(np.hstack((faultsX2D,np.atleast_2d(faultsY).T)),columns=['x','y','target'])
bc2D = pd.DataFrame(np.hstack((bcX2D,np.atleast_2d(bcY).T)),columns=['x','y','target'])

faults2D.to_csv(out+'Faults2D.csv')
bc2D.to_csv(out+'BreastC2D.csv')


