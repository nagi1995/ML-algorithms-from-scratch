# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:19:47 2022

@author: Nagesh
"""
import sys
sys.path.insert(0, '..')
from LinearRegression.linear_regression import CustomPlainLinearRegression, CustomLassoRegression, CustomRidgeRegression, CustomElasticNet
from LogisticRegression.logistic_regression import CustomLogisticRegression, CustomSGDClassifier
from PCA.pca import CustomPCA
from kNN.knn import CustomKNNClassifier, CustomKNNRegressor
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

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
    m = SGDClassifier(loss = 'log', penalty = 'l2', alpha = 1)
    m.fit(x, y)
    print('\nw: ',m.coef_)
    print('b: ', m.intercept_)
    print('prediction: ', m.predict(x[0].reshape(1, -1)))


def test_CustomPCA():
    x = np.random.randn(20, 5)
    custom =  CustomPCA(n_components = 2)
    custom.fit(x)
    
    m, n = x.shape
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)    
    for j in range(n):
        x[:, j] = ( x[:, j] - mean[j] ) / std[j]
    
    
    model = PCA(n_components = 2)
    model.fit(x)
    print("sklearn eigen_values: \n", model.explained_variance_)
    print("custom eigen_values: \n", custom.eigen_values)
    print("sklearn eigen_vectors: \n", model.components_)
    print("custom eigen_vectors: \n", custom.eigen_vectors)
    print("sklearn explained_variance_ratio_: \n", model.explained_variance_ratio_)
    print("custom explained_variance_ratio_: \n", custom.explained_variance_ratio)

def test_CustomKNNClassifier():
    x = np.array([[-1], [-2], [-3], [-4], [-5], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    custom = CustomKNNClassifier(k = 3)
    custom.fit(x, y)
    print("prediction:", custom.predict(np.array([[1.5]])))
    print("prediction:", custom.predict(np.array([[-1.5]])))
    for i in range(10):
        print("prediction:", custom.predict(np.array([[0]])))
        
def test_CustomKNNRegressor():
    x = np.array([[-.1], [-.2], [-.3], [-.4], [-.5], [.1], [.2], [.3], [.4], [.5]])
    y = np.array([-.1, -.2, -.3, -.4, -.5, .1, .2, .3, .4, .5])
    custom = CustomKNNRegressor(k = 3)
    model = KNeighborsRegressor(n_neighbors = 3)
    model.fit(x, y)
    custom.fit(x, y)
    print("custom prediction:", custom.predict(np.array([[.14]])), "sklearn prediction:",  model.predict(np.array([[.14]])))
    print("custom prediction:", custom.predict(np.array([[-.15]])), "sklearn prediction:",  model.predict(np.array([[-.15]])))
    for i in range(10):
        print("custom prediction:", custom.predict(np.array([[0]])), "sklearn prediction:",  model.predict(np.array([[0]])))

if __name__ == '__main__':
    # test_CustomPlainLinearRegression()
    # test_CustomLassoRegression()
    # test_CustomRidgeRegression()
    # test_CustomElasticNet()
    # test_CustomLogisticRegression()
    # test_CustomSGDClassifier()
    # test_CustomPCA()
    test_CustomKNNClassifier()
    test_CustomKNNRegressor()
