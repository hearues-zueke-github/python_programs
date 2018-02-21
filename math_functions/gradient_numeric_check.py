#! /usr/bin/python2.7

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

f = lambda X, A: X.dot(A).dot(X)
f_err = lambda X, A, C: np.sum((f(X, A)-C)**2)/n**2/2.
f_grad = lambda X, A, C: (lambda e: (X.dot(A).T.dot(e)+e.dot(A.dot(X).T))/n**2)(f(X, A)-C)

f2 = lambda X, A, B: X.dot(A).dot(X)+B.dot(X)
f2_err = lambda X, A, B, C: np.sum((f2(X, A, B)-C)**2)/n**2/2.
f2_grad = lambda X, A, B, C: (lambda e: (X.dot(A).T.dot(e)+e.dot(A.dot(X).T)+B.T.dot(e))/n**2)(f2(X, A, B)-C)

f3 = lambda X, A: X.T.dot(A).dot(X)
f3_err = lambda X, A, C: np.sum((f3(X, A)-C)**2)/n**2/2.
f3_grad = lambda X, A, C: (lambda e: (A.dot(X).dot(e.T)+A.T.dot(X).dot(e))/n**2)(f3(X, A)-C)

f4 = lambda X, Y, A: X.dot(A).dot(Y)
f4_err = lambda X, Y, A, C: np.sum((f4(X, Y, A)-C)**2)/n**2/2.
f4_grad = lambda X, Y, A, C: (lambda e: (e.dot(Y.T).dot(A.T)/n**2, A.T.dot(X.T).dot(e)/n**2))(f4(X, Y, A)-C)

def get_numerical_gradient(X, A, C):
    gradient = np.zeros(X.shape)
    epsilon = 0.0001
    
    for j in xrange(0, n):
        for i in xrange(0, n):
            X[j, i] += epsilon
            f1 = f3_err(X, A, C)
            X[j, i] -= epsilon*2
            f2 = f3_err(X, A, C)
            X[j, i] += epsilon
            gradient[j, i] = (f1-f2)/epsilon/2.

    return gradient

def get_numerical_gradient_f4(X, Y, A, C):
    gradient_X = np.zeros(X.shape)
    gradient_Y = np.zeros(X.shape)
    epsilon = 0.0001
    
    for j in xrange(0, n):
        for i in xrange(0, n):
            X[j, i] += epsilon
            f1 = f4_err(X, Y, A, C)
            X[j, i] -= epsilon*2
            f2 = f4_err(X, Y, A, C)
            X[j, i] += epsilon
            gradient_X[j, i] = (f1-f2)/epsilon/2.
            
            Y[j, i] += epsilon
            f1 = f4_err(X, Y, A, C)
            Y[j, i] -= epsilon*2
            f2 = f4_err(X, Y, A, C)
            Y[j, i] += epsilon
            gradient_Y[j, i] = (f1-f2)/epsilon/2.

    return gradient_X, gradient_Y

def get_real_gradient(X, A, C):
    e = f(X, A)-C
    
    gradient = X.dot(A).T.dot(e)+e.dot(A.dot(X).T)

    return gradient/n**2


n = 8
get_random_matrix = lambda: np.random.random((n, n))*2-1

X_1 = get_random_matrix()
Y_1 = get_random_matrix()
A = get_random_matrix()
B = get_random_matrix()
C = X_1.dot(A).dot(Y_1)

m = 20
# Xs = [get_random_matrix() for _ in xrange(0, m)]
# Ys = [get_random_matrix() for _ in xrange(0, m)]
As = [get_random_matrix() for _ in xrange(0, m)]
Cs = [f4(X_1+get_random_matrix()*0.1, Y_1+get_random_matrix()*0.1, Ai+get_random_matrix()*0.1) for Ai in As]

X = get_random_matrix()
Y = get_random_matrix()

print("X:\n{}".format(X))
print("Y:\n{}".format(X))
print("A:\n{}".format(A))
print("C:\n{}".format(C))

Z = f4(X, Y, A)
err = f4_err(X, Y, A, C)

print("Z:\n{}".format(Z))
print("err: {}".format(err))

grad_num_X, grad_num_Y = get_numerical_gradient_f4(X, Y, A, C)
print("grad_num_X:\n{}".format(grad_num_X))
print("grad_num_Y:\n{}".format(grad_num_Y))

grad_real_X, grad_real_Y = f4_grad(X, Y, A, C)
print("grad_real_X:\n{}".format(grad_real_X))
print("grad_real_Y:\n{}".format(grad_real_Y))

raw_input("testing gradients: numeric vs real")

X_best = X.copy()
err_prev = f3_err(X_best, A, C)
eta = 0.002
errors_1 = []
for iteration in xrange(1, 801):
    X_best = X_best - eta*f3_grad(X, A, C)
    err = f3_err(X_best, A, C)
    if err_prev > err:
        eta *= 1.01
        if eta > 5.:
            eta = 5
    else:
        eta *= 0.7
        if eta < 0.0000001:
            eta = 0.0000001
    err_prev = err
    errors_1.append(err)
    if iteration % 100 == 0:
        print("iteration: {}, err: {}, eta: {}".format(iteration, err, eta))

print("")

X_best = X.copy()
Y_best = Y.copy()
err_prev = f4_err(X_best, Y_best, A, C)
eta = 0.002
errors_2 = []
for iteration in xrange(1, 801):
    grad_X_sum, grad_Y_sum = f4_grad(X_best, Y_best, As[0], Cs[0])
    for Ai, Ci in zip(As[1:], Cs[1:]):
        grad_X, grad_Y = f4_grad(X_best, Y_best, Ai, Ci)
        grad_X_sum += grad_X
        grad_Y_sum += grad_Y
    X_best = X_best - eta*grad_X_sum
    Y_best = Y_best - eta*grad_Y_sum
    err = f4_err(X_best, Y_best, A, C)
    if err_prev > err:
        eta *= 1.01
        if eta > 5.:
            eta = 5
    else:
        eta *= 0.7
        if eta < 0.0000001:
            eta = 0.0000001
    err_prev = err
    errors_2.append(err)
    if iteration % 100 == 0:
        print("iteration: {}, err: {}, eta: {}".format(iteration, err, eta))

# print("X_1:\n{}".format(X_1))
# print("X_best:\n{}".format(X_best))
# print("Y_1:\n{}".format(Y_1))
# print("Y_best:\n{}".format(Y_best))

plt.figure()
plt.plot(errors_1, "b-")
plt.plot(errors_2, "g-")
plt.show()
