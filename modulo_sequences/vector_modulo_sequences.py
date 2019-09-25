#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time
from functools import reduce

from PIL import Image

import numpy as np

from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

if __name__ == "__main__":
    ri = np.random.randint
    n = 3
    nn = int(sys.argv[1])
    t1 = (n, )

    # print("x: {}".format(x))
    # print("a: {}".format(a))
    # print("c: {}".format(c))

    lst_m = []
    lst_amount_max_cycles = []
    lst_max_len_cycle = []

    for m in range(1, nn+1):
        print("m: {}".format(m))
        best_xs_vals = []
        best_as = []
        best_cs = []
        best_xs = []
        best_vals = []
        max_len = 0
        for i in range(0, 50000):
            x = ri(0, m, t1)
            a = ri(0, m, t1)
            c = ri(0, m, (1, ))[0]
            x_vals = [tuple(x)]
            while True:
                x_val = (x.dot(a)+c)%m
                new_x = np.roll(x, 1)
                new_x[0] = x_val
                t = tuple(new_x.tolist())
                if t in x_vals:
                    break
                x = new_x
                x_vals.append(t)
            length = len(x_vals)
            if max_len < length:
                max_len = length
                best_vals = [(tuple(a), c)]
                best_xs_vals = [x_vals]
                print("m: {}, i: {}, max_len: {}".format(m, i, max_len))
            elif max_len == length:
                t = (tuple(a), c)
                if not t in best_vals:
                    best_vals.append(t)
                    best_xs_vals.append(x_vals)
                # best_a = a
                # best_c = c
                # best_x = x
                # best_x_vals = x_vals
                    print("m: {}, i: {}, max_len: {}".format(m, i, max_len))
        # print("max_len: {}, best_a: {}, best_c: {}, best_x: {}, best_x_vals: {}".format(max_len, best_a, best_c, best_x, best_x_vals))

        # print("best_vals: {}".format(best_vals))
        # print("len(best_vals): {}".format(len(best_vals)))

        lst_m.append(m)
        lst_amount_max_cycles.append(len(best_vals))
        lst_max_len_cycle.append(max_len)

    print("lst_m: {}".format(lst_m))
    print("lst_amount_max_cycles: {}".format(lst_amount_max_cycles))
    print("lst_max_len_cycle: {}".format(lst_max_len_cycle))
