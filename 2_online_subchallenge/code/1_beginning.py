__author__ = 'Ales'

import numpy as np
import matplotlib.pyplot as plt


import Orange

def cost(X, Y, theta):
    Theta = theta.reshape(Y.shape[1], X.shape[1])
    M = np.dot(X, Theta.T)
    P = np.exp(M - np.max(M, axis=1)[:, None])
    P /= np.sum(P, axis=1)[:, None]
    cost = -np.sum(np.log(P) * Y) / X.shape[0]
    return cost

def grad(X, Y, theta):
    Theta = theta.reshape(Y.shape[1], X.shape[1])
    M = np.dot(X, Theta.T)
    P = np.exp(M - np.max(M, axis=1)[:, None])
    P /= np.sum(P, axis=1)[:, None]
    grad = np.dot(X.T, P - Y).T / X.shape[0]
    return grad.ravel()

def numerical_grad(f, params, epsilon):
    num_grad = np.zeros_like(theta)
    perturb = np.zeros_like(params)
    for i in range(params.size):
        perturb[i] = epsilon
        j1 = f(params + perturb)
        j2 = f(params - perturb)
        num_grad[i] = (j1 - j2) / (2. * epsilon)
        perturb[i] = 0
    return num_grad

data = Orange.data.Table('iris')
X = data.X
Y = np.eye(3)[data.Y.astype(int)]
theta = np.random.randn(3 * 4)

ag = grad(X, Y, theta)
ng = numerical_grad(lambda params: cost(X, Y, params), theta, 1e-4)
print(np.sum((ag - ng)**2))
