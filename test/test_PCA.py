# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:19:47 2022

@author: Nagesh
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PCA.pca import CustomPCA
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA




def test_CustomPCA():
    x = np.random.randn(20, 5)
    custom =  CustomPCA(n_components = 2)
    custom.fit(x)
    
    m, n = x.shape
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)    
    for j in range(n):
        x[:, j] = ( x[:, j] - mean[j] ) / std[j]
    
    
    model = PCA(n_components = 2)
    model.fit(x)
    print("sklearn eigen_values: \n", model.explained_variance_)
    print("custom eigen_values: \n", custom.eigen_values)
    print("sklearn eigen_vectors: \n", model.components_)
    print("custom eigen_vectors: \n", custom.eigen_vectors)
    print("sklearn explained_variance_ratio_: \n", model.explained_variance_ratio_)
    print("custom explained_variance_ratio_: \n", custom.explained_variance_ratio)


test_CustomPCA()

