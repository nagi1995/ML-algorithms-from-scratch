# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:19:47 2022

@author: Nagesh
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LinearRegression.linear_regression import CustomPlainLinearRegression, CustomLassoRegression, CustomRidgeRegression, CustomElasticNet
import numpy as np




def test_CustomPlainLinearRegression():
    x = np.random.randn(1000, 5) 
    y = np.sum(x, axis = 1)
    x += 0.1 * np.random.randn(1000, 5)
    
    model = CustomPlainLinearRegression(lr = 0.1)
    model.fit(x, y)
    print('\nw:', model.w)
    print('b: ', model.b)

def test_CustomLassoRegression():
    x = np.random.randn(1000, 5) 
    y = x[:, 0]
    #x += 0.1 * np.random.randn(1000, 5)
    
    model = CustomLassoRegression(lr = 0.1, alpha = .5)
    model.fit(x, y)
    print('\nw:', model.w)
    print('b: ', model.b)

def test_CustomRidgeRegression():
    x = np.random.randn(1000, 5) 
    y = x[:, 0]
    #x += 0.1 * np.random.randn(1000, 5)
    
    model = CustomRidgeRegression(lr = 0.1, alpha = .5)
    model.fit(x, y)
    print('\nw:', model.w)
    print('b: ', model.b)

def test_CustomElasticNet():
    x = np.random.randn(1000, 5) 
    y = x[:, 0]
    #x += 0.1 * np.random.randn(1000, 5)
    
    model = CustomElasticNet(lr = 0.1, alpha = .5, l1_ratio = .1)
    model.fit(x, y)
    print('\nw:', model.w)
    print('b: ', model.b)

test_CustomPlainLinearRegression()
test_CustomLassoRegression()
test_CustomRidgeRegression()
test_CustomElasticNet()

