#! /usr/bin/python2.7

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

f = lambda v, x: v.dot(x)
f_err = lambda v, x, b: np.sum((f(v, x)-b)**2)/2

def get_grad_1st(v, x, b):
    n = x.shape[0]
    grad = np.zeros((n, ))
    epsilon = 0.0001

    for i in xrange(0, n):
        x[i] += epsilon
        f1 = f_err(v, x, b)
        x[i] -= epsilon*2
        f2 = f_err(v, x, b)
        x[i] += epsilon
        grad[i] = (f1-f2)/2/epsilon

    return grad

def get_grad_2nd(v, x, b):
    n = x.shape[0]
    grad = np.zeros((n, n))
    epsilon = 0.0001

    for j in xrange(0, n):
        for i in xrange(0, n):
            x[j] += epsilon
            x[i] += epsilon
            f11 = f_err(v, x, b)
            x[i] -= epsilon*2
            f12 = f_err(v, x, b)
            x[i] += epsilon
            f1 = (f11-f12)/2/epsilon
            
            x[j] -= epsilon*2
            x[i] += epsilon
            f21 = f_err(v, x, b)
            x[i] -= epsilon*2
            f22 = f_err(v, x, b)
            x[i] += epsilon
            f2 = (f21-f22)/2/epsilon
            
            x[j] += epsilon

            grad[j, i] = (f1-f2)/2/epsilon

    return grad

n = 5
get_random_vector = lambda: np.random.random((n, ))*2-1

v = get_random_vector()
x = get_random_vector()
b = np.random.random()*2-1
print("v: {}".format(v))
print("x: {}".format(x))
print("b: {}".format(b))

grad = get_grad_1st(v, x, b)
print("grad:\n{}".format(grad))

grad_2 = get_grad_2nd(v, x, b)
print("grad_2:\n{}".format(grad_2))

grad_real = v*(f(v, x)-b)
print("grad_real:\n{}".format(grad_real))

grad_real_2 = np.outer(v, v)
print("grad_real_2:\n{}".format(grad_real_2))

x_best = x.copy()
