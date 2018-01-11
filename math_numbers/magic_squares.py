#! /usr/bin/python3.5

import numpy as np

rows = 3
b = np.zeros((30000, rows*2+2)).astype(np.int)
for i in range(0, b.shape[0]):
    a = np.random.randint(1, 20, (rows, rows)).astype(np.int)
    # print("a:\n{}".format(a))

    b[i] = np.hstack((np.sum(a, axis=0), np.sum(a, axis=1), np.sum(a.diagonal()), np.sum(a[np.arange(0, a.shape[0]), np.arange(a.shape[0]-1, -1, -1)])))
    # print("sums: {}".format(sums))

c = np.all(b[:, 0] == b[:, 1:].T, axis=0)
print("b:\n{}".format(b))
print("c: {}".format(c))
print("np.where(c): {}".format(np.where(c)[0]))
