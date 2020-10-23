#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import io
import datetime
import os
import pathlib
import re
import shutil
import string
import subprocess
import sys
import time
import mmap

import numpy as np
import multiprocessing as mp

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

sys.path.append('../combinatorics')
from different_combinations import get_all_combinations_repeat

if __name__ == '__main__':
    n = int(sys.argv[1])
    base = 3

    print("n: {}, base: {}".format(n, base))

    max_cycle_len = 0
    best_factors = []
    l_missing_factors = []

    # a_c = 1
    # a_l = 1
    # a_r = 1
    # a_u = 1
    # a_d = 1
    amount = 0

    for a_c in range(0, base):
     for a_l in range(0, base):
      for a_r in range(0, base):
       for a_u in range(0, base):
        for a_d in range(0, base):
            factors = (a_c, a_l, a_r, a_u, a_d, )
            # print("factors: {}".format(factors))

            arr = np.zeros((n, n), dtype=np.uint8)
            arr[0, 0] = 1
            
            l_arr = [arr]

            # print("k: {}".format(0))
            # print("- arr:\n{}".format(arr))

            for k in range(1, 1000):
                arr_l = np.roll(arr, 1, axis=1)
                arr_r = np.roll(arr, -1, axis=1)
                arr_u = np.roll(arr, 1, axis=0)
                arr_d = np.roll(arr, -1, axis=0)
                
                arr_new = (a_c*arr + a_l*arr_l + a_r*arr_r + a_u*arr_u + a_d*arr_d) % base

                l_arr.append(arr_new)
                arr = arr_new

                # print("k: {}".format(k))
                # print("- arr:\n{}".format(arr))

            k_last = len(l_arr)-1
            arr_last = l_arr[-1]
            is_found_same = False
            for k, arr_k in zip(range(k_last-1, -1, -1), reversed(l_arr[:-1])):
                if np.all(arr_last==arr_k):
                    # print('Found same arr_k at:')
                    # print("k: {}".format(k))
                    # print("arr_k:\n{}".format(arr_k))
                    is_found_same = True
                    break

            del l_arr

            if is_found_same:
                cycle_len = k_last-k
                if max_cycle_len < cycle_len:
                    max_cycle_len = cycle_len
                    best_factors = [factors]
                elif max_cycle_len == cycle_len:
                    best_factors.append(factors)
            else:
                l_missing_factors.append(factors)
            # print("max_cycle_len: {}".format(max_cycle_len))

            # amount += 1
            # if amount % 100 == 0:
            #     print("amount: {}, max_cycle_len: {}".format(amount, max_cycle_len))

    print("    max_cycle_len: {}".format(max_cycle_len))
    print("    len(best_factors): {}".format(len(best_factors)))
    print("    l_missing_factors: {}".format(l_missing_factors))

"""
some values:
for the canonical array with the form:
   [[1,0,...,0], [0,0,...,0], ..., [0,0,...0]]

const n = 3
n: 3, base: 2
    max_cycle_len: 3
    len(best_factors): 24
n: 3, base: 3
    max_cycle_len: 6
    len(best_factors): 80
n: 3, base: 4
    max_cycle_len: 6
    len(best_factors): 752
n: 3, base: 5
    max_cycle_len: 24
    len(best_factors): 2720

const base = 3
n: 2, base: 3
    max_cycle_len: 2
    len(best_factors): 189
n: 3, base: 3
    max_cycle_len: 6
    len(best_factors): 80
n: 4, base: 3
    max_cycle_len: 8
    len(best_factors): 200
n: 5, base: 3
    max_cycle_len: 80
    len(best_factors): 184
n: 6, base: 3
    max_cycle_len: 6
    len(best_factors): 188
n: 7, base: 3
    max_cycle_len: 728
    len(best_factors): 168
n: 8, base: 3
    max_cycle_len: 8
    len(best_factors): 232
n: 9, base: 3
    max_cycle_len: 18
    len(best_factors): 80
"""

    # base = 10
    # d = {}
    # n_max = 100
    # for y in range(0, n_max):
    #     for x in range(0, n_max):
    #         if y == 0:
    #             v_y = 0
    #         else:
    #             v_y = d[(y-1, x)]

    #         if x == 0:
    #             v_x = 0
    #         else:
    #             v_x = d[(y, x-1)]

    #         # formula is: v = (vx + vy**2 + 1) % base
    #         d[(y, x)] = (v_x + v_y**2 + 1) % base

    # # print("d: {}".format(d))

    # arr = np.zeros((n_max, n_max), dtype=np.int)
    # for y in range(0, n_max):
    #     for x in range(0, n_max):
    #         arr[y, x] = d[(y, x)]

    # print("arr:\n{}".format(arr))
