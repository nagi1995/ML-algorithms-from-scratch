# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:19:47 2022

@author: Nagesh
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LogisticRegression.logistic_regression import CustomLogisticRegression, CustomSGDClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier


def test_CustomLogisticRegression():
    x, y = make_classification(n_samples = 100000, n_features = 5, 
                               n_informative = 3, n_redundant = 2, 
                               n_classes = 2, random_state = 15)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    model = CustomLogisticRegression(lr = .1, alpha = 1, penalty = 'l2')
    model.fit(x, y)
    print('\nw: ', model.w)
    print('b: ', model.b)
    print('prediction: ', model.predict(x[0]))
    m = LogisticRegression(C = 1, penalty = 'l2')
    m.fit(x, y)
    print('\nw: ',m.coef_)
    print('b: ', m.intercept_)
    print('prediction: ', m.predict(x[0].reshape(1, -1)))

def test_CustomSGDClassifier():
    x, y = make_classification(n_samples = 1000, n_features = 5, 
                               n_informative = 3, n_redundant = 2, 
                               n_classes = 2, random_state = 150)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    model = CustomSGDClassifier(lr = 0.0001, alpha = 1, penalty = 'l2', max_iter = 1000)
    model.fit(x, y)
    print('\nw: ', model.w)
    print('b: ', model.b)
    print('prediction: ', model.predict(x[0]))
    m = SGDClassifier(loss = 'log_loss', penalty = 'l2', alpha = 1)
    m.fit(x, y)
    print('\nw: ',m.coef_)
    print('b: ', m.intercept_)
    print('prediction: ', m.predict(x[0].reshape(1, -1)))

test_CustomLogisticRegression()
test_CustomSGDClassifier()
