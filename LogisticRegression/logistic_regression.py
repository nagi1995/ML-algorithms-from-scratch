# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:17:26 2022

@author: Nagesh
"""

import numpy as np
from tqdm import tqdm
import random

class CustomLogisticRegression():
    
    def __init__(self, lr, alpha, penalty, l1_ratio = None, max_iter = 100):
        self.lr = lr
        self.alpha = alpha
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        if self.penalty not in ['l1', 'l2', 'elasticnet']:
            raise Exception('penalty can either be \'l1\' or \'l2\' or \'elasticnet\'')
        if self.penalty == 'elasticnet' and self.l1_ratio is None:
            raise Exception('When penalty is \'elasticnet\', \'l1_ratio\' should be specified')
        self.max_iter = max_iter
    
    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z)) # shape: (m, 1)
    
    def _fx(self, w, x, b):
        fx = np.dot(x, w) + b
        return self._sigmoid( fx ) # shape: (m, 1)
    
    def _gradient_mod_w(self, w):
        grad_mod_w = []
        for i in range(len(w)):
            if w[i] > 0:
                grad_mod_w.append(1)
            elif w[i] < 0:
                grad_mod_w.append(-1)
            else:
                grad_mod_w.append(random.choice([-1, 1]))
        return np.array(grad_mod_w).reshape(-1, 1) # shape: (n, 1)
    
    def _gradient(self, x, y, w, b, m):
        fx = self._fx(w, x, b)
        if self.penalty == 'l1':
            dw = -( np.dot(x.T, y - fx ) - ( self.alpha * self._gradient_mod_w(w) * 0.5 ) ) / m # shape: (n, 1)
        elif self.penalty == 'l2':
            dw = -( np.dot(x.T, y - fx ) - ( self.alpha * w ) ) / m # shape: (n, 1)
        elif self.penalty == 'elasticnet':
            dw = -( np.dot(x.T, y - fx ) - ( self.alpha * ( w * ( 1 - self.l1_ratio ) + ( self._gradient_mod_w(w) * self.l1_ratio * 0.5 ) ) ) ) / m # shape: (n, 1)
        else:
            pass
        db = -( sum(y - fx) ) / m # shape: (1, 1)
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
            
        self.w = w
        self.b = b
    def predict(self, x):
        fx = self._fx(self.w, x, self.b)
        if fx.shape[0] > 1:
            y_pred = []
            fx = np.ravel(fx)
            for i in range(len(fx)):
                if fx[i] > .5:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            return np.array(y_pred)
        else:
            if fx > .5:
                return 1
            else:
                return 0


class CustomSGDClassifier():
    def __init__(self, lr, alpha, penalty, l1_ratio = None, max_iter = 100):
        self.lr = lr
        self.alpha = alpha
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        if self.penalty not in ['l1', 'l2', 'elasticnet']:
            raise Exception('penalty can either be \'l1\' or \'l2\' or \'elasticnet\'')
        if self.penalty == 'elasticnet' and self.l1_ratio is None:
            raise Exception('When penalty is \'elasticnet\', \'l1_ratio\' should be specified')
        self.max_iter = max_iter
    
    def _sigmoid(self, z):
        return 1 / ( 1 + np.exp(-z) ) # shape: (1, 1)
    
    def _fx(self, w, x, b):
        fx = np.dot(x, w) + b
        return self._sigmoid( fx ) # shape: (1, 1)
    
    def _gradient_mod_w(self, w):
        if w > 0:
            return 1
        elif w < 0:
            return -1
        else:
            return random.choice([-1, 1])
    
    def _gradient(self, x, y, w, b, n):
        fx = self._fx(w, x, b)
        dw = np.zeros_like(x)
        if self.penalty == 'l1':
            for i in range(n):
                dw[i] = -( ( x[i] * (y - fx) ) - ( self.alpha * self._gradient_mod_w(w[i]) ) / n ) # shape: (1, 1)
        elif self.penalty == 'l2':
            for i in range(n):
                dw[i] = -( ( x[i] * (y - fx) ) - ( self.alpha * w[i] ) / n ) # shape: (1, 1)
        elif self.penalty == 'elasticnet':
            for i in range(n):
                dw[i] = -( ( x[i] * (y - fx) ) - ( self.alpha * ( w * ( 1 - self.l1_ratio ) + ( self._gradient_mod_w(w) * self.l1_ratio * 0.5 ) ) ) / n )  # shape: (1, 1)
        else:
            pass
        db = -( y - fx ) # shape: (1, 1)
        return dw, db
    
    def fit(self, x, y):
        if len(np.array(x).shape) != 2:
            raise Exception("Pass x in 2-D format")
        m, n = x.shape # shape of x: (m, n)
        w = np.zeros((n,)) # shape of w: (n,)
        
        b = 0
        y = y.reshape(-1, 1) # shape: (m, 1)
        for i in tqdm(range(self.max_iter)):
            for j in range(n):
                dw, db = self._gradient(x[j], y[j], w, b, n)
                # print('shape of dw: ', dw.shape)
                # print('shape of w: ', w.shape)
                w -= self.lr * dw
                b -= self.lr * db
        self.w = w
        self.b = b
    def predict(self, x):
        fx = self._fx(self.w, x, self.b)
        if fx.shape[0] > 1:
            y_pred = []
            fx = np.ravel(fx)
            for i in range(len(fx)):
                if fx[i] > .5:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            return np.array(y_pred)
        else:
            if fx > .5:
                return 1
            else:
                return 0