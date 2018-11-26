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

# Needed for lowering the exponents!
exp_positions = np.array([0, 2, 5, 7])
def get_random_f(n):
    arr = np.random.randint(0, n, (10, ))
    exps = np.random.randint(0, 4, (len(exp_positions), ))
    arr[exp_positions] = exps

    s = "lambda v: np.array([v[0]**{}*{}+v[1]**{}*{}+{}, v[0]**{}*{}+v[1]**{}*{}+{}]) % {}".format(
        *[arr[i] for i in range(0, 10)],
        # arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9], 
        n)

    return arr, eval(s)

def t():
    pass

if __name__ == "__main__":
    n = 10
    amount = np.zeros((n, n), dtype=np.int)

    v = np.array([0, 0])

    best_lens = []
    best_arr = []
    max_len = 0
    for i in range(0, 1000):
        if i % 100 == 0:
            print("i: {}".format(i))
        arr, f = get_random_f(n)

        amount[v[0], v[1]] += 1

        l = 0
        v_start = v.copy()
        for _ in range(0, n**2):
            v = f(v)
            amount[v[0], v[1]] += 1
            l += 1

            if np.all(v_start==v):
                break

        if l == n**2:
            best_lens.append(l)
            best_arr.append(arr)

    best_arr = np.array(best_arr)
    print("best_arr:\n{}".format(best_arr))
    # print("best_arr:")
    # for arr in best_arr:
    #     print("arr: {}".format(arr))
    print("len(best_arr): {}".format(len(best_arr)))
