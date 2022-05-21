# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:19:47 2022

@author: Nagesh
"""
import sys
sys.path.insert(0, '..')
from LinearRegression.linear_regression import PlainLinearRegression, LassoRegression, RidgeRegression, ElasticNet
import numpy as np


def test_PlainLinearRegression():
    x = np.random.randn(1000, 5) 
    y = np.sum(x, axis = 1)
    x += 0.1 * np.random.randn(1000, 5)
    
    model = PlainLinearRegression(lr = 0.1)
    model.fit(x, y)
    print('\nw:', model.w)
    print('b: ', model.b)
    
def test_LassoRegression():
    x = np.random.randn(1000, 5) 
    y = x[:, 0]
    #x += 0.1 * np.random.randn(1000, 5)
    
    model = LassoRegression(lr = 0.1, alpha = .5)
    model.fit(x, y)
    print('\nw:', model.w)
    print('b: ', model.b)

def test_RidgeRegression():
    x = np.random.randn(1000, 5) 
    y = x[:, 0]
    #x += 0.1 * np.random.randn(1000, 5)
    
    model = RidgeRegression(lr = 0.1, alpha = .5)
    model.fit(x, y)
    print('\nw:', model.w)
    print('b: ', model.b)

def test_ElasticNet():
    x = np.random.randn(1000, 5) 
    y = x[:, 0]
    #x += 0.1 * np.random.randn(1000, 5)
    
    model = ElasticNet(lr = 0.1, alpha = .5, l1_ratio = .1)
    model.fit(x, y)
    print('\nw:', model.w)
    print('b: ', model.b)
    
    
if __name__ == '__main__':
    test_PlainLinearRegression()
    test_LassoRegression()
    test_RidgeRegression()
    test_ElasticNet()
