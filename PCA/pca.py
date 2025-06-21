# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:27:41 2022

@author: Nagesh
"""

import numpy as np

class CustomPCA():
    
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, x):
        x = np.array(x)
        if len(x.shape) < 2:
            raise Exception("Input should be a 2-d array or a list of list")
        m, n = x.shape
        mean = np.mean(x, axis = 0)
        std = np.std(x, axis = 0)    
        for j in range(n):
            x[:, j] = ( x[:, j] - mean[j] ) / std[j]
        
        cov = np.cov(x.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        sorted_indices = np.argsort(eigen_values)[::-1]
        self.eigen_values = eigen_values[sorted_indices][: self.n_components]
        self.eigen_vectors = eigen_vectors[:, sorted_indices][:, :self.n_components].T
        self.explained_variance_ratio = self.eigen_values/sum(eigen_values)
    
    def transform(self, x):
        return np.dot(x, self.eigen_vectors.T)
    
        

