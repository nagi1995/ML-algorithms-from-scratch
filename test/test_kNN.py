# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:19:47 2022

@author: Nagesh
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kNN.knn import CustomKNNClassifier, CustomKNNRegressor
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsRegressor


def test_CustomKNNClassifier():
    x = np.array([[-1], [-2], [-3], [-4], [-5], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    custom = CustomKNNClassifier(k = 3)
    custom.fit(x, y)
    print("prediction:", custom.predict(np.array([[1.5]])))
    print("prediction:", custom.predict(np.array([[-1.5]])))
    for i in range(10):
        print("prediction:", custom.predict(np.array([[0]])))
        
def test_CustomKNNRegressor():
    x = np.array([[-.1], [-.2], [-.3], [-.4], [-.5], [.1], [.2], [.3], [.4], [.5]])
    y = np.array([-.1, -.2, -.3, -.4, -.5, .1, .2, .3, .4, .5])
    custom = CustomKNNRegressor(k = 3)
    model = KNeighborsRegressor(n_neighbors = 3)
    model.fit(x, y)
    custom.fit(x, y)
    print("custom prediction:", custom.predict(np.array([[.14]])), "sklearn prediction:",  model.predict(np.array([[.14]])))
    print("custom prediction:", custom.predict(np.array([[-.15]])), "sklearn prediction:",  model.predict(np.array([[-.15]])))
    for i in range(10):
        print("custom prediction:", custom.predict(np.array([[0]])), "sklearn prediction:",  model.predict(np.array([[0]])))

test_CustomKNNClassifier()
test_CustomKNNRegressor()
