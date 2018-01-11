#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np

np.set_printoptions(threshold=np.nan)

ps = np.array([2, 3, 5, 7]).astype("object") #np.uint64)
# ps = np.array([2, 3, 5, 7]).astype(int) #np.uint64)

def get_potence_table(n, m):
    potence_table = np.zeros((n**m, m)).astype("object") #np.uint64)
    # potence_table = np.zeros((n**m, m)).astype(int) #np.uint64)
    potence_table[:n, 0] = np.arange(n)

    for i in xrange(1, m):
        n_i = n**i
        for j in xrange(1, n):
            potence_table[n_i*j:n_i*(j+1), :i] = potence_table[:n_i, :i]
            potence_table[n_i*j:n_i*(j+1), i] = j

    return potence_table

print("ps: {}".format(ps))

m = ps.shape[0]
n = 12

potence_table = get_potence_table(n, m)

print("potence_table:\n{}".format(potence_table))

print("ps**potence_table: {}".format(ps**potence_table))

numbers = np.sort(np.prod(ps**potence_table, axis=1))

print("numbers: {}".format(numbers))

prod = 1
for p in ps:
    prod *= int(p)**(n-1)

print("max prod: {}".format(prod))
