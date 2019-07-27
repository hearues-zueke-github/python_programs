#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import pdb
import sys

# Needed for DataFrame (databases etc.)
import numpy as np
import pandas as pd
import pandas.io.sql as sql

from functools import reduce

from dotmap import DotMap

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

USER_HOME_PATH = os.path.expanduser('~')+'/'
print("USER_HOME_PATH: {}".format(USER_HOME_PATH))

def get_self_rotated_matrix(arr):
    arrs = [np.roll(arr, i) for i in range(0, arr.shape[0])]
    return np.array(arrs)


def log_int(n, m):
    s = 0
    while (n >= m):n //= m;s += 1
    return s


def get_explicit_rotated_value(n, m):
    if n < m:return n
    ni = n
    while ni > m-1:b_val = log_int(log_int(ni, m), 2);mi = m**(2**b_val);ni = (ni-ni//mi)%mi
    return ni


def get_self_rotated_array(nn, m):
    return np.array([get_explicit_rotated_value(j, m) for j in range(0, nn)])


def get_absolute_vals_rotated_values(m, times):
    v = np.arange(0, m)
    for i in range(0, times):
        print("m: {}, i: {}".format(m, i))
        k = m**2**i
        a = get_self_rotated_matrix(v)
        v = (a+np.arange(0, k).reshape((-1, 1))*k).reshape((-1, ))
    return v


def get_pretty_printed_evaluation(n, m):
    print("Example a({}):".format(n))
    if n < m:
        print("a({}) = {}.".format(n, n))
        return n
    ni = n
    while ni > m-1:
        ni_prev = ni
        b_val = log_int(log_int(ni, m), 2)
        mi = m**(2**b_val)
        ni = (ni-ni//mi)%mi
        print("b({ni_prev}) = {b_val} -> a({ni_prev}) = a(({ni_prev} - [{ni_prev} / {m}^(2^{b_val})]) mod {m}^(2^{b_val})) = a({ni})".format(
            ni_prev=ni_prev, b_val=b_val, ni=ni, m=m), end="")
        if ni < m:
            print(" = {ni}.".format(ni=ni))
            print("Therefore a({n}) = {ni}.".format(n=n, ni=ni))
        else:
            print(".")

    return ni

if __name__ == "__main__":
    # for m in range(2, 3):

    nn = int(sys.argv[1])

    m = nn
    arr = np.arange(0, m).astype(np.uint8)
    # print("i: {}, arr: {}".format(0, arr))
    for i in range(1, 4):
        arr = get_self_rotated_matrix(arr).reshape((-1, ))
        # print("i: {}, arr.shape: {}".format(i, arr.shape))
        # print("arr.reshape((-1, int(np.sqrt(arr.shape[0])))):\n{}".format(arr.reshape((-1, int(np.sqrt(arr.shape[0]))))))
        # arr2 = np.array([get_explicit_rotated_value(j, m) for j in range(0, arr.shape[0])])
        arr2 = get_self_rotated_array(arr.shape[0], m)
        # print("arr2.reshape((-1, int(np.sqrt(arr2.shape[0])))):\n{}".format(arr2.reshape((-1, int(np.sqrt(arr2.shape[0]))))))

        assert(np.all(arr==arr2))

    # np.sum(np.vstack((arr[:-1], arr[1:])).T*m**np.arange(length_digits-1, -1, -1), axis=1)

    print("m: {}, arr[:75]: {}".format(m, (arr[:75]).tolist()))
    length_digits = 2
    len_arr = arr.shape[0]
    arr_digits = np.vstack((arr[i:len_arr-length_digits+1+i] for i in range(0, length_digits))).T
    arr_digits_sum = np.sum(arr_digits*m**np.arange(length_digits-1, -1, -1), axis=1)
    print("length_digits: {}".format(length_digits))
    print("arr_digits_sum[:30]: {}".format(arr_digits_sum[:30]))

    values_pairwise = np.sum(arr.reshape((-1, m))*m**np.arange(m-1, -1, -1), axis=1)
    print("values_pairwise.tolist(): {}".format(values_pairwise.tolist()))

    row_size = m
    values_per_row_1 = np.sum(arr[:row_size*row_size].reshape((-1, row_size))*m**np.arange(row_size-1, -1, -1), axis=1)
    row_size = m**2
    values_per_row_2 = np.sum(arr[:row_size*row_size].reshape((-1, row_size))*m**np.arange(row_size-1, -1, -1), axis=1)

    values_per_row = np.hstack((values_per_row_1, values_per_row_2))
    print("values_per_row.tolist(): {}".format(values_per_row.tolist()))

    # print("arr: {}".format(arr))

    # length = arr.shape[0]
    # best_r = 0

    # most_similar_numbers = 0
    # for r in range(1, length):
    #     amount_similar = np.sum(arr==np.roll(arr, r))
    #     if most_similar_numbers < amount_similar:
    #         print("amount_similar: {}, r: {}".format(amount_similar, r))
    #         most_similar_numbers = amount_similar
    #         best_r = r

    # print("most_similar_numbers: {}, best_r: {}".format(most_similar_numbers, best_r))

    # print("arr[:30].tolist(): {}".format(arr[:30].tolist()))

    # x=np.hstack(((0, ), np.where(arr[1:]!=arr[:-1])[0]+1, (arr.shape[0], )))
    # u, c = np.unique(x[1:]-x[:-1], return_counts=True)
    # print("u: {}".format(u))
    # print("c: {}".format(c))