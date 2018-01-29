#! /usr/bin/python2.7

import sys

import numpy as np

m = 5
modulo = 3

a = np.random.randint(0, 2, (m, ))
b = np.random.randint(0, 2, (m, ))

C = np.add.outer(a, b) % modulo

an1 = np.dot(C, a) % modulo
bn1 = np.dot(C, b) % modulo

an2 = np.dot(a, C) % modulo
bn2 = np.dot(b, C) % modulo

print("a:\n{}".format(a))
print("b:\n{}".format(b))
print("C:\n{}".format(C))
print("an1:\n{}".format(an1))
print("bn1:\n{}".format(bn1))
print("an2:\n{}".format(an2))
print("bn2:\n{}".format(bn2))
