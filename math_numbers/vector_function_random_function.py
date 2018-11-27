#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import os
import string
import sys

import multiprocessing as mp
import numpy as np

from dotmap import DotMap
from multiprocessing import Process, Queue

from time import time

def t():
    pass

if __name__ == "__main__":
    n = 10
    amount = np.zeros((n, n), dtype=np.int)

    amount_functions = 1000
    amount_args = 10

    exp_positions = np.array([0, 2, 5, 7])

    f_str_template = "lambda *v: np.array([v[0]**{}*{}+v[1]**{}*{}+{}, v[0]**{}*{}+v[1]**{}*{}+{}]) % {}"

    # first create all function arguments!
    args = np.random.randint(0, n, (amount_functions, 10))
    exp_args = np.random.randint(0, 5, (amount_functions, len(exp_positions)))
    args[:, exp_positions] = exp_args
    fs = np.apply_along_axis(lambda x: eval(f_str_template.format(*x, n)), -1, args).reshape((-1, 1))

    vfs = np.vectorize(fs)

    sys.exit(-1)

    # TODO: change this a little bit more efficient!
    # Needed for lowering the exponents!
    def get_random_f(n):
        arr = np.random.randint(0, n, (10, ))
        exps = np.random.randint(0, 4, (len(exp_positions), ))
        arr[exp_positions] = exps

        s = "lambda v: np.array([v[0]**{}*{}+v[1]**{}*{}+{}, v[0]**{}*{}+v[1]**{}*{}+{}]) % {}".format(
            *[arr[i] for i in range(0, 10)], n)

        return arr, eval(s)

    best_arr = []
    for i in range(0, 1000):
        if i % 100 == 0:
            print("i: {}".format(i))
        arr, f = get_random_f(n)

        amount[v[0], v[1]] += 1

        l = 0
        v = np.array([0, 0])
        
        v_start = v.copy()
        for _ in range(0, n**2):
            v = f(v)
            amount[v[0], v[1]] += 1
            l += 1

            if np.all(v_start==v):
                break

        if l == n**2:
            best_arr.append(arr)

    best_arr = np.array(best_arr)
    print("best_arr:\n{}".format(best_arr))
    # print("best_arr:")
    # for arr in best_arr:
    #     print("arr: {}".format(arr))
    print("len(best_arr): {}".format(len(best_arr)))
