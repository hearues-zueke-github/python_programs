#! /usr/bin/python2.7

import sys

import numpy as np

n = 5
A = np.random.randint(0, 2, (n, n))
B = np.random.randint(0, 2, (n, n))
b = np.random.randint(0, 2, (n, ))

print("A:\n{}".format(A))
print("B:\n{}".format(B))
print("b:\n{}".format(b))

C = np.dot(A, B) % 2
print("C:\n{}".format(C))

c = np.dot(A, b) % 2
print("c:\n{}".format(c))

def get_binary_X(bits):
    X = np.zeros((2**bits, bits)).astype(np.int)
    X[1, 0] = 1
    for i in xrange(1, bits):
        X[2**i:2**(i+1), :i] = X[:2**i, :i]
        X[2**i:2**(i+1), i] = 1

    return X

def find_universal_matrix_A(n):
    X = get_binary_X(n)
    # print("X:\n{}".format(X))

    while True:
        A = np.random.randint(0, 2, (n, n))
        
        Y = np.dot(X, A) % 2
        numbers = np.sum(Y*2**np.arange(0, n), axis=1)
        numbers_sorted = np.sort(numbers)
        diff = numbers_sorted[1:]-numbers_sorted[:-1]
        sum_diff = np.sum(diff!=1)

        if sum_diff == 0:
            break

    return A

A = find_universal_matrix_A(n)
print("A:\n{}".format(A))

A = np.random.random((n, n))*2-1
X = np.random.random((30, n))*2-1

print("A:\n{}".format(A))
print("X:\n{}".format(X))

Y = np.dot(X, A)
print("Y:\n{}".format(Y))

Y_round = Y.copy().astype(np.int)
Y_round[Y<0.] = 0
Y_round[Y>=0.] = 1

print("Y_round:\n{}".format(Y_round))

def save_file_data(file_name, X, T):
    length = X.shape[0]
    idx_1 = int(length*0.6)
    idx_2 = int(length*0.8)
    X_train = X[:idx_1]
    T_train = T[:idx_1]
    X_valid = X[idx_1:idx_2]
    T_valid = T[idx_1:idx_2]
    X_test = X[idx_2:]
    T_test = T[idx_2:]

    with open(file_name, "wb") as f:
        np.savez(f, X_train=X_train, T_train=T_train,
                    X_valid=X_valid, T_valid=T_valid,
                    X_test=X_test, T_test=T_test)

file_name = "matrix_mutliply_data.npz"
save_file_data(file_name, X, Y)

file_name = "matrix_mutliply_data_rounded.npz"
save_file_data(file_name, X, Y_round)
