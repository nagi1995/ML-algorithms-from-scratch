# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:01:17 2022

@author: Nagesh
"""

import numpy as np
import pandas as pd

class CustomKNNClassifier():
    
    def __init__(self, k):
        self.k = k
    
    def fit(self, x, y):
        if len(np.array(x).shape) != 2:
            raise Exception("Pass x in 2-D format")
        self.x_train = x
        self.y_train = y
    
    def predict_one(self, x):
        distances = []
        df = pd.DataFrame()
        df['labels'] = self.y_train
        for j in range(self.x_train.shape[0]):
            distances.append(np.linalg.norm(x - self.x_train[j]))
        df['distances'] = distances
        df2 = df.sort_values(by = ['distances']).head(self.k)
        k_labels = df2['labels'].tolist()
        predicted_label = max(set(k_labels), key = k_labels.count)
        return predicted_label
        
    
    def predict(self, x_test):
        if len(np.array(x_test).shape) != 2:
            raise Exception("Pass input in 2-D format")
        predicted_labels = []
        for i in range(x_test.shape[0]):
            predicted_labels.append(self.predict_one(x_test[i]))
        return predicted_labels



class CustomKNNRegressor():
    
    def __init__(self, k):
        self.k = k
    
    def fit(self, x, y):
        if len(np.array(x).shape) != 2:
            raise Exception("Pass x in 2-D format")
        self.x_train = x
        self.y_train = y
    
    def predict_one(self, x):
        distances = []
        df = pd.DataFrame()
        df['y'] = self.y_train
        for j in range(self.x_train.shape[0]):
            distances.append(np.linalg.norm(x - self.x_train[j]))
        df['distances'] = distances
        df2 = df.sort_values(by = ['distances']).head(self.k)
        prediction = df2['y'].mean()
        return prediction
        
    
    def predict(self, x_test):
        if len(np.array(x_test).shape) != 2:
            raise Exception("Pass input in 2-D format")
        predictions = []
        for i in range(x_test.shape[0]):
            predictions.append(self.predict_one(x_test[i]))
        return predictions
