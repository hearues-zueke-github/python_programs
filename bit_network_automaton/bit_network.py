#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import traceback

import numpy as np
import pandas as pd

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile

sys.path.append('..')
from utils import mkdirs
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
sys.path.append('../combinatorics')
import different_combinations

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

def calc_cycle_length(l_idx, l):
    d = {tuple(l.tolist()): 0}

    i = 1
    while True:
        l = np.bitwise_xor.reduce(l[l_idx], axis=1)
        t = tuple(l.tolist())
        if t in d:
            d['last'] = (t, i)
            break
        d[t] = i
        i += 1

    t_first, i_last = d['last']
    i_first = d[t_first]
    cycle_length = i_last - i_first
    return t_first, cycle_length


if __name__ == '__main__':
    # TODO: define function for checking all inital bits and idx_rows for cycle lengths!

    # s_cycle_length = set()

    # n = int(sys.argv[1])
    # # n = 8
    # # m = 1
    # for m in range(1, n+1):
    #     print("n: {}, m: {}".format(n, m))
    #     arr_combs = different_combinations.get_all_permutations_increment(n, m)
    #     # arr_combs_idx = different_combinations.get_all_combinations_repeat(arr_combs.shape[0], n)

    #     arr_bits = different_combinations.get_all_combinations_repeat(2, n)

    #     from math import factorial as fac
    #     amount_initals = 2**n*(fac(n)//fac(m)//fac(n-m))**n

    #     rows_combs = arr_combs.shape[0]
    #     # rows_combs_idx = arr_combs_idx.shape[0]
    #     # rows_bits = arr_bits.shape[0]
    #     # print("- amount_initals: {}".format(amount_initals))
    #     # print("- rows_combs_idx: {}".format(rows_combs_idx))
    #     # print("- rows_bits: {}".format(rows_bits))
    #     # assert rows_combs_idx*rows_bits == amount_initals

    #     for _ in range(0, 1000):
    #     # for i_combs_idx in np.random.randint(0, arr_combs_idx.shape[0], (1000, )):
    #     # for combs_idx in arr_combs_idx:
    #         combs_idx = np.random.randint(0, rows_combs, (n, ))
    #         # combs_idx = arr_combs_idx[i_combs_idx]
    #         l_idx = arr_combs[combs_idx]
    #         for l in arr_bits:
    #             t_first, cycle_length = calc_cycle_length(l_idx, l)
    #             if cycle_length not in s_cycle_length:
    #                 s_cycle_length.add(cycle_length)
    #     print("- s_cycle_length: {}".format(s_cycle_length))
    #     print("- len(s_cycle_length): {}".format(len(s_cycle_length)))
    # print()
    # print("final: s_cycle_length: {}".format(s_cycle_length))
    # print("final: len(s_cycle_length): {}".format(len(s_cycle_length)))

    # sys.exit()

    obj_d_s_cycle_length_file_path = OBJS_DIR_PATH+'d_s_cycle_length.pkl.gz'
    def create_d_s_cycle_length():
        return {}

    d_s_cycle_length = get_pkl_gz_obj(create_d_s_cycle_length, obj_d_s_cycle_length_file_path)
    n_max = 12
    for n in range(1, n_max+1):
        for m in range(1, n+1):
            print("n: {}, m: {}".format(n, m))

            arr_combs = different_combinations.get_all_permutations_increment(n, m)

            t = (n, m)
            if t not in d_s_cycle_length:
                d_s_cycle_length[t] = set()
            s_cycle_length = d_s_cycle_length[t]

            d_best_vals = {
                'l_idx': None,
                'l': None,
                'cycle_length': 0,
            }

            d_cycle_length = {}
            max_cycle_length = 0
            for i in range(0, 1000):
                # l_idx = np.array([np.random.permutation(np.arange(0, n))[:m] for _ in range(0, n)])
                # l_idx = np.sort(l_idx, axis=1)
                # l_idx = l_idx[np.argsort(np.sum(l_idx*n**np.arange(m-1, -1, -1), axis=1))]
                l_idx = arr_combs[np.random.randint(0, arr_combs.shape[0], (n, ))]
                # l_idx = arr_combs[np.sort(np.random.randint(0, arr_combs.shape[0], (n, )))]
                l = np.random.randint(0, 2, (n, ))
                t_first, cycle_length = calc_cycle_length(l_idx, l)
                if t_first not in d_cycle_length:
                    d_cycle_length[t_first] = cycle_length
                if cycle_length not in s_cycle_length:
                    s_cycle_length.add(cycle_length)

                if max_cycle_length < cycle_length:
                    d_best_vals['cycle_length'] = cycle_length
                    d_best_vals['l_idx'] = l_idx
                    d_best_vals['l'] = l
                    max_cycle_length = cycle_length
                    print("i: {:6}, max_cycle_length: {}".format(i, max_cycle_length))
                    print("- t_first: {}, cycle_length: {}".format(t_first, cycle_length))
            if n == 10 and m == 3:
                sys.exit()
            print("sorted(s_cycle_length): {}".format(sorted(s_cycle_length)))
            print("len(s_cycle_length): {}".format(len(s_cycle_length)))

    l_t_sorted = sorted([(k, len(v)) for k, v in d_s_cycle_length.items()])
    print("l_t_sorted: {}".format(l_t_sorted))

    l_max = []
    d = {}
    for ni in range(1, n_max+1):
        l_t = [((ni, mi), d_s_cycle_length[(ni, mi)]) for mi in range(1, ni+1)]
        l_t_len = [(ni, mi, len(s)) for (ni, mi), s in l_t]
        i_argmax = np.argmax(np.array(l_t_len)[:, 2])
        print("i_argmax: {}".format(i_argmax))
        l_max.append(l_t_len[i_argmax])
        d[ni] = l_t

    save_pkl_gz_obj(d_s_cycle_length, obj_d_s_cycle_length_file_path)
