# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:01:17 2022

@author: Nagesh
"""

import numpy as np
from tqdm import tqdm


class CustomSVMClassifier():

    def __init__(self, lr = 0.001, C = 1, n_iters = 1000):
        self.lr = lr
        self.C = C
        self.n_iters = n_iters 

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        n_samples, n_features = x.shape
        y_ = np.where(y <= 0, -1, 1) # transform labels to [-1, 1]

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in tqdm(range(self.n_iters)):
            for idx, xi in enumerate(x):
                if y_[idx] * (np.dot(xi, self.w) + self.b) >= 1:
                    dw = 2 * self.w
                    db = 0
                else:
                    dw = 2 * self.w - self.C * y_[idx] * xi 
                    db = -self.C * y_[idx]
                
                self.w -= self.lr * dw 
                self.b -= self.lr * db 

    def predict(self, x):
        x = np.array(x)
        raw_output = np.dot(x, self.w) + self.b 
        return np.where(raw_output >= 0, 1, 0)



class CustomSVMRegressor():
    def __init__(self, lr = 0.001, C = 1.0, epsilon = 0.1, n_iters = 1000):
        self.lr = lr 
        self.C = C
        self.epsilon = epsilon
        self.n_iters = n_iters

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)

        n_samples, n_features = x.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in tqdm(range(self.n_iters)):
            for idx, xi in enumerate(x):
                yi = y[idx]

                y_pred = np.dot(self.w, xi) + self.b
                error = y_pred - yi
                if abs(error) <= self.epsilon:
                    dw = 2*self.w
                    db = 0
                elif error > self.epsilon:
                    dw = 2 * self.w + self.C * xi
                    db = self.C
                else: # error < -epsilon
                    dw = 2 * self.w - self.C * xi
                    db = -self.C
                
                self.w -= self.lr * dw
                self.b -= self.lr * db
    

    def predict(self, x):
        x = np.array(x)
        return np.dot(x, self.w) + self.b
