#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import operator

import numpy as np

from copy import deepcopy

import sys
sys.path.append("../encryption")

import Utils

if __name__ == "__main__":
    max_p = 6
    n = 2**max_p # amount of elements = modulo = number base

    # factors = np.random.randint(1, n, (lines, 2)).astype(object)

    # factors = np.vstack((np.random.randint(0, n, (lines, )),
    #                      np.random.randint(0, n//2, (lines, ))*2+1))
    
    factors = np.array([[2, 9, 6, 0],
                        [2, 9, 6, 2],
                        [2, 9, 6, 4],
                        [2, 9, 12, 0],
                        [2, 9, 12, 2],
                        [2, 9, 12, 4]]).T

    # lines = 1000
    # a_0 = np.random.randint(0, n, (lines, ))
    # a_1 = np.random.randint(0, n//2, (lines, ))*2+1
    # factors = np.vstack((a_0, a_1))
    # for p in range(2, max_p):
    #     a_p = np.random.randint(0, n//2**p, (lines, ))*2
    #     factors = np.vstack((factors, a_p))
    # a_2 = np.random.randint(0, n//2, (lines, ))*2
    # a_3 = np.random.randint(0, n//4, (lines, ))*2
    # a_4 = np.random.randint(0, n//8, (lines, ))*2
    # a_5 = np.random.randint(0, n//16, (lines, ))*2
    
    # factors = np.vstack((a_0, a_1, a_2, a_3, a_4))                     
    # factors = np.vstack((a_0, a_1, a_2, a_3, a_4))                     

    # print("factors: {}".format(factors))

    def apply_linear_function(x, n, factors):
        for a0, a1 in factors:
            x = (a0+a1*x) % n
        return int(x)

    def apply_linear_functions(x, n, factors):
        fs_x = np.zeros((x.shape[0], factors.shape[1]), dtype=np.int)+factors[0]
        for i, a in zip(np.arange(1, factors.shape[0]+1), factors[1:]):
            fs_x += (a*x**i) % n
        return fs_x % n

    # x = np.random.randint(0, n)
    x = np.arange(0, n).reshape((-1, 1))
    # print("x: {}".format(x))

    fs_x = apply_linear_functions(x, n, factors).T

    # print("fs_x:\n{}".format(fs_x))

    unique_length = []
    # check unique
    for f_x in fs_x:
        unique_length.append(np.unique(f_x).shape[0])

    not_same_length = np.sum(np.array(unique_length)!=n)
    # print("unique_length: {}".format(unique_length))
    print("not_same_length: {}".format(not_same_length))

    if not_same_length > 0:
        print("Not the same length!")
        sys.exit(-5)


