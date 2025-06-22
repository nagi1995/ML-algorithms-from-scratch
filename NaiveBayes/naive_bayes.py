# -*- coding: utf-8 -*-
"""
Created on Sun May 15 21:54:47 2022

@author: Nagesh
"""

import numpy as np
from tqdm import tqdm
import random

class CustomNaiveBayes():

    def __init__(self, laplace = 0):
        self.laplace = laplace
        self.class_probs = {}
        self.feature_probs = {}
        self.classes = None 
    
    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        n_samples, n_features = x.shape
        self.classes = np.unique(y).tolist()

        class_counts = {c: 0 for c in self.classes}
        feature_counts = {c: np.zeros(n_features) for c in self.classes}

        for xi, yi in tqdm(zip(x, y)):
            class_counts[yi] =+ 1
            feature_counts[yi] += xi
        
        self.class_probs = {c: class_counts[c]/n_samples for c in self.classes}

        self.feature_probs = {}

        for c in self.classes:
            counts = feature_counts[c]
            total = np.sum(counts)
            probs = (counts + self.laplace) / (total + (self.laplace * n_features))
            self.feature_probs[c] = probs.tolist()
    
    def predict(self, x):
        x = np.array(x)
        predictions = []

        for xi in tqdm(x):
            class_scores = {}
            for c in self.classes:
                log_prob = np.log(self.class_probs[c])
                probs = np.array(self.feature_probs[c])
                log_prob += np.sum(xi*np.log(probs))
                class_scores[c] = log_prob
            predictions.append(max(class_scores, key = class_scores.get))
        
        return predictions