#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gmpy2
import marshal
import os
import pickle
import string

import multiprocessing as mp
import numpy as np

from dotmap import DotMap
from multiprocessing import Process, Queue

from time import time

get_two_number_array = lambda arr: np.vstack((arr[:-1], arr[1:])).T

get_many_stacked_array = lambda arr, n, base: np.sum(np.vstack((arr[i:-n+i] for i in range(0, n))).T*base**np.arange(n-1, -1, -1), axis=-1)

if __name__ == "__main__":
    num = 32**50000
    base = 14

    all_symbols = string.digits+string.ascii_lowercase+string.ascii_uppercase
    symbol_to_int = {s: i for i, s in enumerate(all_symbols)}
    num_b = gmpy2.digits(num, base)
    arr = np.array(list(map(lambda x: symbol_to_int[x], num_b)))
    arr_sequence = get_two_number_array(get_many_stacked_array(arr, 1, base))

    arr_sequence_comb = np.sum(arr_sequence*base**np.arange(1, -1, -1), axis=-1)
    arr_sequence_comb_uniq_comb = np.unique(arr_sequence_comb, return_counts=True)

    amount = np.zeros((base, base), dtype=np.int)
    for idx, a in np.array(arr_sequence_comb_uniq_comb).T:
        amount[idx//base, idx%base] = a

    print("amount:\n{}".format(amount))
    max_val = np.max(amount)
    min_val = (lambda x: max_val if x < 1 else x)(np.min(amount))
    print("min_val: {}".format(min_val))
    print("max_val: {}".format(max_val))

    get_row_col = lambda x: (x//base, x%base)
    print("np.argmin(amount): {}".format(get_row_col(np.argmin(amount))))
    print("np.argmax(amount): {}".format(get_row_col(np.argmax(amount))))

    print("max_val/min_val: {}".format(max_val/min_val))

    mean_all = np.mean(amount)
    mean_x = np.mean(amount, axis=1)
    mean_y = np.mean(amount, axis=0)

    print("mean_all: {}".format(mean_all))
    print("mean_x: {}".format(mean_x))
    print("mean_y: {}".format(mean_y))

    amount_1_dig = np.zeros((base, ), dtype=np.int)
    arr_unique, arr_counts = np.unique(arr, return_counts=True)
    for idx, a in zip(arr_unique, arr_counts):
        amount_1_dig[idx] = a
    # print("arr_unique:\n{}".format(arr_unique))
    # print("arr_counts:\n{}".format(arr_counts))
    print("amount_1_dig\n {}".format(amount_1_dig))

    mean_1_dig = np.mean(amount_1_dig)
    std_1_dig = np.std(amount_1_dig)
    print("mean_1_dig: {}".format(mean_1_dig))
    print("std_1_dig: {}".format(std_1_dig))
