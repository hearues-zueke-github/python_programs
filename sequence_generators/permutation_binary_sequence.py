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

sys.path.append('../combinatorics')
import different_combinations

import numpy as np
import multiprocessing as mp

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

from math import factorial as fac

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

if __name__ == '__main__':
    get_perm = different_combinations.get_permutation_table
    get_perm_inc = different_combinations.get_all_permutations_increment
    get_perm_inc_gen = different_combinations.get_permutation_table_generator
    
    # for n1, n2 in [(3, 4), (2, 6), (4, 4), (3, 6)]:
    #     n = n1 + n2
        
    #     l_arr = []
    #     for arr in get_perm_inc_gen(n1, n2):
    #         l_arr.append(arr.copy())

    #     arr_perm_n = np.vstack(l_arr)
    #     arr_perm_n_sum_row = np.sort(np.sum(arr_perm_n.astype(object)*[2**i for i in range(n-1, -1, -1)], axis=1))

    #     arr_perm_orig = get_perm(n)
    #     arr_perm_orig_sum_row = np.sort(np.sum(arr_perm_orig.astype(object)*[2**i for i in range(n-1, -1, -1)], axis=1))

    #     assert np.all(arr_perm_n_sum_row == arr_perm_orig_sum_row)

    # sys.exit()

    l_max_c = [(1, 0)]
    l_max_c_c = [(1, 0)]
    d_d_u_c = {}
    d_d_cu_c = {}
    # for n in range(12, 13):
    for n in range(15, 16):
    
    # for n in range(3, 10):
    # for n in range(7, 12):
    # for n in range(2, 9):
        print("\nn: {}".format(n))

        # arr = get_perm(n)
        
        if n < 5:
            n1 = 1
        else:
            n1 = 5

        n2 = n - n1

        # n1 = n // 2
        # n2 = n - n1

        d_u_c = {}
        # d_c = {}
        d_cu_c = {}
        # d_c_c = {}

        for num_arr, arr in enumerate(get_perm_inc_gen(n1, n2)):
            print("- n: {}, n1: {}, num_arr: {}".format(n, n1, num_arr))
            # # with loop!
            # arr0 = np.hstack((arr, arr[:, :1]))
            # # del arr
            # arr = arr0

            # TODO 2020.07.16: find the next values for the n = [12, 13, 14, ...]!

            arr_bin = (arr[:, :-1]<arr[:, 1:]).astype(np.uint8).reshape((-1, )).view(','.join(['u1']*(arr.shape[1]-1)))
            del arr
            u, c = np.unique(arr_bin, return_counts=True)
            del arr_bin
            c_u, c_c = np.unique(c, return_counts=True)
            

            # globals()['u'] = u
            # globals()['c'] = c
            for a, i in zip(u, c):
                t = tuple(a.tolist())
                if t not in d_u_c:
                    d_u_c[t] = i
                    continue
                d_u_c[t] += i

            for a, i in zip(c_u, c_c):
                if a not in d_cu_c:
                    d_cu_c[a] = i
                    continue
                d_cu_c[a] += i

            # print("u: {}".format(u))
            # print("c: {}".format(c))
            # print("len(c): {}".format(len(c)))
            # print("np.max(c): {}".format(np.max(c)))
        
        l_max_c.append((n, max(list(d_u_c.values()))))
        l_max_c_c.append((n, max(list(d_cu_c.values()))))

        print("l_max_c[-1]: {}".format(l_max_c[-1]))
        print("l_max_c_c[-1]: {}".format(l_max_c_c[-1]))

        # d_arr_u[n] = np.array(u)
        # d_arr_c[n] = np.array(c)

        # d_arr_c_u[n] = np.array(c_u)
        # d_arr_c_c[n] = np.array(c_c)

        d_d_u_c[n] = d_u_c
        d_d_cu_c[n] = d_cu_c

        # del u
        # del c
        # del c_u
        # del c_c

    print("l_max_c: {}".format(l_max_c))
    l_c = [num_c for _, num_c in l_max_c]
    print("l_c: {}".format(l_c))

    print("l_max_c_c: {}".format(l_max_c_c))
    l_c_c = [num_c_c for _, num_c_c in l_max_c_c]
    print("l_c_c: {}".format(l_c_c))

# permutations:
    # With NO loop!
    # n    = [1, 2, 3, 4,  5,  6,   7,    8,    9] 
    # a(n) = [0, 1, 2, 5, 16, 61, 272, 1385, 7936]

    # l_max_c: [(1, 0), (2, 1), (3, 2), (4, 5), (5, 16), (6, 61), (7, 272), (8, 1385)]
    # l_c: [0, 1, 2, 5, 16, 61, 272, 1385]
    # l_max_c_c: [(1, 0), (2, 2), (3, 2), (4, 4), (5, 4), (6, 4), (7, 6), (8, 8)]
    # l_c_c: [0, 2, 2, 4, 4, 4, 6, 8]

    # WITH loop!
    # n    = [1, 2, 3, 4, 5,  6,   7,    8,    9,    10,     11] 
    # a(n) = [0, 1, 1, 4, 8, 48, 136, 1088, 3968, 39680, 176896]

    # l_max_c: [(1, 0), (2, 1), (3, 1), (4, 4), (5, 8), (6, 48), (7, 136), (8, 1088), (9, 3968), (10, 39680), (11, 176896)]
    # l_c: [0, 1, 1, 4, 8, 48, 136, 1088, 3968, 39680, 176896]
    # l_max_c_c: [(1, 0), (2, 2), (3, 6), (4, 8), (5, 10), (6, 12), (7, 28), (8, 32), (9, 36), (10, 40), (11, 44)]
    # l_c_c: [0, 2, 6, 8, 10, 12, 28, 32, 36, 40, 44]

    # [d_c_c[i].shape[0] for i in range(2, 10)]
    # [1, 1, 3, 3, 7, 8, 17, 22, 43, 62]


# with repeat:
    # WITH loop!
    # l_max_c: [(1, 0), (2, 2), (3, 7), (4, 34), (5, 278), (6, 2703), (7, 27485), (8, 408068)]
    # l_c: [0, 2, 7, 34, 278, 2703, 27485, 408068]
    # l_max_c_c: [(1, 0), (2, 2), (3, 3), (4, 4), (5, 5), (6, 12), (7, 14), (8, 16)]
    # l_c_c: [0, 2, 3, 4, 5, 12, 14, 16]

    # [d_arr_c_c[i].shape[0] for i in range(2, 9)]
    # [2, 3, 5, 7, 12, 17, 29]
