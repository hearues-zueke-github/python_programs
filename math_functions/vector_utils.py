#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

import numpy as np

from math import factorial as fac

import decimal
decimal.getcontext().prec = 800
from decimal import Decimal as Dec
Decimal = Dec

import matplotlib.pyplot as plt

def inv_with_decimal(A):
    shape = A.shape
    A = np.array([Dec(int(x)) for x in A.reshape((-1, ))]).reshape(shape)
    # print("as dec A:\n{}".format(A))
    # print("A.dtype:\n{}".format(A.dtype))
    # print("A[2, 0]: {}".format(A[2, 0]+Dec(1)/3))

    I = np.zeros(shape, dtype=np.int)
    I[(np.arange(0, shape[0]), np.arange(0, shape[1]))] = 1
    I = np.array([Dec(int(x)) for x in I.reshape((-1, ))]).reshape(shape)

    # print("I:\n{}".format(I))

    X = np.hstack((A, I))

    line_0 = X[0].copy()
    X[0] = X[1]
    X[1] = line_0
    # print("X:\n{}".format(X))

    # X[1] -= X[0]*X[1, 0]/X[0, 0]
    # X[2] -= X[0]*X[2, 0]/X[0, 0]

    # X[2] -= X[1]*X[2, 1]/X[1, 1]

    # X[1] -= X[2]*X[1, 2]/X[2, 2]
    # X[0] -= X[2]*X[0, 2]/X[2, 2]
    
    # X[0] -= X[1]*X[0, 1]/X[1, 1]

    epsilon = Decimal("0.0000000000000000000000000000000000000001")

    # permutations_done = []
    for i in range(0, shape[0]-1):
        for j in range(i+1, shape[0]):
            # print("up-down: i: {}, j: {}".format(i, j))

            # if np.abs(X[i, i]) <= epsilon:
            #     for k in range(0, shape[0]):
            #         if k == i or np.abs(X[k, i]) <= epsilon:
            #             continue
            #         permutations_done.append((i, k))
            #         print("permutation: i: {}, k: {}".format(i, k))
            #         line_k = X[k].copy()
            #         X[k] = X[i]
            #         X[i] = line_k
            line = X[j]-X[i]*X[j, i]/X[i, i]

            X[j] = line
            if np.abs(line[j]) <= epsilon:
                print("X[{}, {}] is close or is 0!".format(i, j))

                # X[j, j] += 1
                # X[j, j+shape[0]] += 1
                # for k in range(j+1, shape[0]):
                #     if k == i or np.abs(X[k, i]) <= epsilon:
                #         continue
                #     # permutations_done.append((i, k))
                #     print("permutation: i: {}, k: {}".format(i, k))
                #     line_k = X[k].copy()
                #     X[k] = X[i]
                #     X[i] = line_k 

            # if len(permutations_done) > 0:
            #     k, l = permutations_done.pop()
            #     line_k = X[k].copy()
            #     X[k] = X[l]
            #     X[l] = line_k

        # print("X[:shape[0]]:\n{}".format(X[:, :shape[0]]))

    for i in range(shape[0]-1, 0, -1):
        for j in range(i-1, -1, -1):
            # print("down-up: i: {}, j: {}".format(i, j))

            # if np.abs(X[i, i]) <= epsilon:
            #     for k in range(0, shape[0]):
            #         if k == i or np.abs(X[k, i]) <= epsilon:
            #             continue
            #         # permutations_done.append((i, k))
            #         print("permutation: i: {}, k: {}".format(i, k))
            #         line_k = X[k].copy()
            #         X[k] = X[i]
            #         X[i] = line_k
            X[j] -= X[i]*X[j, i]/X[i, i]

            # if len(permutations_done) > 0:
            #     k, l = permutations_done.pop()
            #     line_k = X[k].copy()
            #     X[k] = X[l]
            #     X[l] = line_k

    # for i, j in permutations_done:
    #     line_i = X[i]
    #     X[i] = X[j]
    #     X[j] = line_i
            
    # X[1] -= X[0]*X[1, 0]/X[0, 0]
    # X[2] -= X[0]*X[2, 0]/X[0, 0]

    # X[2] -= X[1]*X[2, 1]/X[1, 1]

    # X[1] -= X[2]*X[1, 2]/X[2, 2]
    # X[0] -= X[2]*X[0, 2]/X[2, 2]
    
    # X[0] -= X[1]*X[0, 1]/X[1, 1]
    X_diag = np.diag(X).reshape((-1, 1))
    # print("X_diag: {}".format(X_diag))
    
    # print("X:\n{}".format(X))

    X /= X_diag
    A_inv = X[:, shape[0]:]

    # print("X:\n{}".format(X))
    # print("A_inv:\n{}".format(A_inv))

    return A_inv


if __name__ == "__main__":
    A = np.array([[2, 3, 4, 5], [3, 4, 5, 1], [3, 6, 7, 9], [3, 2, 4, 3]])
    n = 4
    # A = np.arange(1, n**2+1).reshape((n, n))
    # A[n-1, n-1] = n**2+10
    print("A:\n{}".format(A))

    A_inv = np.linalg.inv(A)
    print("A_inv:\n{}".format(A_inv))

    A_inv_dec = inv_with_decimal(A)
