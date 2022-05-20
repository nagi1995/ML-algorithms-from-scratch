# -*- coding: utf-8 -*-
"""
Created on Sun May 15 21:54:47 2022

@author: Nagesh
"""

import numpy as np
from tqdm import tqdm


class LinearRegressionClass():
    
    def __init__(self, lr,  max_iter = 100):
        self.lr = lr
        self.max_iter = max_iter
    
    def _fx(self, w, x, b):
        return np.dot(x, w) + b # shape: (m, 1)
        
    def _gradient(self, x, y, w, b, m):
        fx = self._fx(w, x, b)
        dw = -( 2 * np.dot(x.T, y - fx) ) / m # shape: (n, 1)
        db = -( 2 * sum(y - fx) ) / m # shape: (1, 1)
        return dw, db
        
        
    def fit(self, x, y):
        if len(np.array(x).shape) != 2:
            raise Exception("Pass x in 2-D format")
        m, n = x.shape # shape of x: (m, n)
        w = np.zeros((n, 1)) # shape of w: (n, 1)
        
        b = 0
        y = y.reshape(-1, 1) # shape: (m, 1)
        for i in tqdm(range(self.max_iter)):
            dw, db = self._gradient(x, y, w, b, m)
            w -= self.lr * dw
            b -= self.lr * db
            #print(w.shape, b.shape)
        self.w = w
        self.b = b
    
    def predict(self, x):
        return self._fx(self.w, x, self.b)

