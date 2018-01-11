#! /usr/bin/python2.7

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

np.set_printoptions(threshold=np.nan)

m = 32
n = 16

a = np.random.randint(0, m, (n, ))
# A = np.random.randint(0, m, (n, n))
# A = np.diag(np.random.permutation(np.arange(0, n)))
idxs = np.random.permutation(np.arange(0, n)).tolist()
A = np.diag(np.ones((n, )).astype(np.int))
A = A[idxs]
print("A:\n{}".format(A))
x = np.random.randint(0, m, (n, ))
X = np.random.randint(0, m, (n, n))

# print("A:\n{}".format(A))

# sys.exit(0)

# print("a:\n{}".format(a))
# print("A:\n{}".format(A))
# print("x:\n{}".format(x))
# print("X:\n{}".format(X))

numbers = []

home = os.path.expanduser("~")
full_path = home+"/Pictures/sequence_matrices/m_{}_n_{}".format(m, n)
if not os.path.exists(full_path):
    os.makedirs(full_path)
os.chdir(full_path)

colors = np.zeros((m, 3))
colors[:] = np.arange(0, m).reshape((-1, 1))
colors = (colors * (256. / m)).astype(np.uint8)

# x = np.dot(X, x) % m
# x = np.dot(A, x) % m
# v = np.dot(a, x) % m

for i in xrange(0, 100):
    X = np.dot(A, X) % m
    # print("i: {}, X:\n{}".format(i, X))
    
    numbers.append(np.sum(X) % m)

    Image.fromarray(colors[X]).save("i_{}.png".format(i), "PNG")

    # sys.exit(0)

print("numbers: {}".format(numbers))


