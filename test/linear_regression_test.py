# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:19:47 2022

@author: Nagesh
"""
import sys
sys.path.insert(0, '..')
from LinearRegression.linear_regression import LinearRegressionClass
import numpy as np

x = np.random.randn(100, 2)
y = np.sum(x, axis = 1)

model = LinearRegressionClass(lr = 0.01)
model.fit(x, y)
print()
print(model.w, model.b)
