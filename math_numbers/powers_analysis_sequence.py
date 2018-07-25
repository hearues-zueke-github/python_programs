#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import marshal
import pickle
import os

import multiprocessing as mp
import numpy as np

from dotmap import DotMap
from multiprocessing import Process, Queue

from time import time

get_two_number_array = lambda arr: np.vstack((arr[:-1], arr[1:])).T

get_many_stacked_array = lambda arr, n: (lambda n: np.sum(np.vstack((arr[i:-n+i] for i in range(0, n))).T*10**np.arange(n-1, -1, -1), axis=-1))(n)

if __name__ == "__main__":
    base = 10
    # arr = np.random.randint(0, base, (600000, ))
    # TODO: make a own C function for getting the digits of a number in a base as a list
    # TODO: or find a pre defined function, with the same behaviour!
    arr = np.array(list(map(int, str(4**1000000))))
    arr_sequence = get_two_number_array(get_many_stacked_array(arr, 1))

    # amount = np.zeros((base, base), dtype=np.int)

    arr_sequence_comb = np.sum(arr_sequence*base**np.arange(1, -1, -1), axis=-1)
    arr_sequence_comb_uniq_comb = np.unique(arr_sequence_comb, return_counts=True)

    amount = arr_sequence_comb_uniq_comb[1].reshape((base, base))

    # for i, j in arr_sequence:
    #     amount[i, j] += 1

    print("amount:\n{}".format(amount))
    min_val = np.min(amount)
    max_val = np.max(amount)
    print("min_val: {}".format(min_val))
    print("max_val: {}".format(max_val))

    print("np.argmin(amount): {}".format(np.argmin(amount)))
    print("np.argmax(amount): {}".format(np.argmax(amount)))

    print("max_val/min_val: {}".format(max_val/min_val))

    arr_unique, arr_counts = np.unique(arr, return_counts=True)
    print("arr_unique:\n{}".format(arr_unique))
    print("arr_counts:\n{}".format(arr_counts))
