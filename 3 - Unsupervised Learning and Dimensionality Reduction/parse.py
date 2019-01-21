# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:39:27 2017

@author: jtay
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
import os 
import sklearn.model_selection as ms

#Make directories if not created
for d in ['BASE','RP','PCA','ICA','RF']:
    n = './Output/{}/{}/'.format(d,d)
    if not os.path.exists(n):
        os.makedirs(n)

#Set output folder
OUT = './Output/BASE/'

## Convert datasets into training and test into HDF output ##

#For faults dataset
train_f = pd.read_csv('./Datasets/train_faults.csv')
test_f = pd.read_csv('./Datasets/test_faults.csv')

train_f.to_hdf(OUT+'datasets.hdf','train_faults',complib='blosc',complevel=9)
test_f.to_hdf(OUT+'datasets.hdf','test_faults',complib='blosc',complevel=9)

#For Breast Cancer dataset
train_bc = pd.read_csv('./Datasets/train_bc.csv')
test_bc = pd.read_csv('./Datasets/test_bc.csv')
train_bc = train_bc.drop(['id'],1)
test_bc = test_bc.drop(['id'],1)

train_bc.to_hdf(OUT+'datasets.hdf','train_bc',complib='blosc',complevel=9)
test_bc.to_hdf(OUT+'datasets.hdf','test_bc',complib='blosc',complevel=9)