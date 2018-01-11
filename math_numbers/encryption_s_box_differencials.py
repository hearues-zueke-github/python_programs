#! /usr/bin/python3

import numpy as np

bits = 3
size = 1 << bits
p = np.random.permutation(np.arange(size)).astype(np.int)

print("p = {}".format(p))

table = np.zeros((size, size)).astype(np.int)

for y in range(0, size):
    for x in range(0, size):
        table[x^y, p[x]^p[y]] += 1

print("table =\n{}".format(table))
