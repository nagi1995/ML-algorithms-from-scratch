# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:19:47 2022

@author: Nagesh
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from NaiveBayes.naive_bayes import CustomNaiveBayes
import numpy as np
from sklearn.datasets import make_classification


def test_CustomNaiveBayes():
    x = [[1,1], [1, 0], [0,1], [0,0], [1,1], [0,1]]
    x_test = [[1,1], [0,0]]
    y = ['yes', 'yes', 'no', 'no', 'yes', 'no']
    model = CustomNaiveBayes(laplace = 1)
    model.fit(x, y)
    print("Feature Probabilities:", model.feature_probs)
    print("Predictions:", model.predict(x_test))

test_CustomNaiveBayes()

