# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:19:47 2022

@author: Nagesh
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SVM.svm import CustomSVMClassifier, CustomSVMRegressor
import numpy as np
from matplotlib import pyplot as plt

def plot_svm(X, y, model):
    X = np.array(X)
    y = np.array(y)
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, model.w, model.b, 0)
    x1_2 = get_hyperplane_value(x0_2, model.w, model.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, model.w, model.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, model.w, model.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, model.w, model.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, model.w, model.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k--')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k--')

    ax.set_ylim(np.amin(X[:, 1]) - 1, np.amax(X[:, 1]) + 1)
    plt.show()


def test_CustomSVMClassifier():
    x = [[-.9, 1e-5], [-.8, 1e-5], [-.1, 1e-5], [.1, 1e-5], [.8, 1e-5], [.9, 1e-5]]
    y = [0, 0, 0, 1, 1, 1]

    model = CustomSVMClassifier()
    model.fit(x, y)

    print(f'w: {model.w}')
    print(f'b: {model.b}')
    x_test = [[-.7, 0], [-.05, 0], [0, 0], [.05, 0], [.7, 0], [-1e-10, 0]]
    print(f'x_test: {x_test}')
    print(f'predictions: {model.predict(x_test)}')
    plot_svm(x, y, model)


def test_CustomSVMRegressor():
    x = [[-.9, 1e-5], [-.8, 1e-5], [-.1, 1e-5], [.1, 1e-5], [.8, 1e-5], [.9, 1e-5]]
    y = [-.9, -0.8, -0.1, 0.1, 0.8, .9] 
    model = CustomSVMRegressor()
    model.fit(x, y)

    print(f'w: {model.w}')
    print(f'b: {model.b}')
    x_test = [[-.7, 0], [-.05, 0], [0, 0], [.05, 0], [.7, 0], [-1e-10, 0]]
    print(f'x_test: {x_test}')
    print(f'predictions: {model.predict(x_test)}')

test_CustomSVMClassifier()
test_CustomSVMRegressor()
