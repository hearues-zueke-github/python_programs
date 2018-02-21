#! /usr/bin/python2.7

import os
import sys

import numpy as np

def get_binary_X(bits):
    X = np.zeros((2**bits, bits)).astype(np.int)
    X[1, 0] = 1
    for i in xrange(1, bits):
        X[2**i:2**(i+1), :i] = X[:2**i, :i]
        X[2**i:2**(i+1), i] = 1

    return X

def find_universal_matrix_A(n):
    X = get_binary_X(n)

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

def get_new_X_T(n, m, length):
    A = np.random.random((n, m))*2.-1.
    X = np.random.random((length, n))

    T = np.dot(X*2.-1., A)
    T_sig = 1 / (1+np.exp(-T))

    T_round = T.copy().astype(np.int)
    T_round[T<0.] = 0
    T_round[T>=0.] = 1

    return X, T_sig, T_round

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

def load_file_data(file_name):
    data = np.load(file_name)
    X_train = data["X_train"]
    T_train = data["T_train"]
    X_valid = data["X_valid"]
    T_valid = data["T_valid"]
    X_test = data["X_test"]
    T_test = data["T_test"]

    print("X_train[:10]:\n{}".format(X_train[:10]))
    print("T_train[:10]:\n{}".format(T_train[:10]))

    print("X.shape: {}".format(X.shape))
    print("T.shape: {}".format(T.shape))

home = os.path.expanduser("~")
matrix_multiply_data = home+"/Documents/matrix_multiply_data"

if not os.path.exists(matrix_multiply_data):
    os.makedirs(matrix_multiply_data)
os.chdir(matrix_multiply_data)

if __name__ == "__main__":
    get_file_name = lambda n, m: "matrix_multiply_data_n_{}_m_{}.npz".format(n, m)
    get_file_name_rounded = lambda n, m: "matrix_multiply_data_n_{}_m_{}_rounded.npz".format(n, m)
    m = 3
    for n in xrange(3, 9):
        print("data_size: n: {}".format(n))
        # X, T, _ = get_new_X_T(n, m, 5000)
        X, T, T_round = get_new_X_T(n, m, 5000)

        file_name = get_file_name(n, m)
        save_file_data(file_name, X, T)

        file_name = get_file_name_rounded(n, m)
        save_file_data(file_name, X, T_round)

    file_name = get_file_name(3, m)
    load_file_data(file_name)
