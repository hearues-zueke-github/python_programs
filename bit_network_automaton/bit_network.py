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

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':
    obj_d_s_cycle_length_file_path = OBJS_DIR_PATH+'d_s_cycle_length.pkl.gz'
    def create_d_s_cycle_length():
        return {}

    d_s_cycle_length = get_pkl_gz_obj(create_d_s_cycle_length, obj_d_s_cycle_length_file_path)

    for n in range(2, 10):
    # n = 3
        for m in range(1, n):
            print("n: {}, m: {}".format(n, m))

            # m = 3
            # l = np.random.randint(0, 2, (n, ))
            # l_idx = np.array([np.random.permutation(np.arange(0, n))[:m] for _ in range(0, n)])
            # arr = np.random.permutation(np.tile(np.arange(0, n), n).reshape((n, n)), axis=1)
            # print("l_idx:\n{}".format(l_idx))

            # t = tuple(l.tolist())
            # print("{}: {}".format(t, 0))

            t = (n, m)
            if t not in d_s_cycle_length:
                d_s_cycle_length[t] = set()
            s_cycle_length = d_s_cycle_length[t]

            def calc_cycle_length(l_idx, l):
                d = {tuple(l.tolist()): 0}

                i = 1
                while True:
                    l = np.bitwise_xor.reduce(l[l_idx], axis=1)
                    t = tuple(l.tolist())
                    # print("{}: {}".format(t, i))
                    if t in d:
                        d['last'] = (t, i)
                        break
                    d[t] = i
                    i += 1
                # l_next = np.bitwise_xor.reduce(l[l_idx], axis=1)

                t_first, i_last = d['last']
                i_first = d[t_first]
                cycle_length = i_last - i_first
                return t_first, cycle_length

            d_cycle_length = {}
            # s_cycle_length = set()
            max_cycle_length = 0
            for i in range(0, 20000):
                l_idx = np.array([np.random.permutation(np.arange(0, n))[:m] for _ in range(0, n)])
                l_idx = np.sort(l_idx, axis=1)
                l_idx = l_idx[np.argsort(np.sum(l_idx*n**np.arange(m-1, -1, -1), axis=1))]
                l = np.random.randint(0, 2, (n, ))
                t_first, cycle_length = calc_cycle_length(l_idx, l)
                if t_first not in d_cycle_length:
                    # print("l: {}".format(l))
                    # print("- t_first: {}, cycle_length: {}".format(t_first, cycle_length))
                    d_cycle_length[t_first] = cycle_length
                if cycle_length not in s_cycle_length:
                    s_cycle_length.add(cycle_length)

                if max_cycle_length < cycle_length:
                    max_cycle_length = cycle_length
                    print("i: {:6}, max_cycle_length: {}".format(i, max_cycle_length))
                    print("- t_first: {}, cycle_length: {}".format(t_first, cycle_length))

            print("s_cycle_length: {}".format(s_cycle_length))
            print("len(s_cycle_length): {}".format(len(s_cycle_length)))

    l_t_sorted = sorted([(k, len(v)) for k, v in d_s_cycle_length.items()])
    print("l_t_sorted: {}".format(l_t_sorted))

    # print("l: {}".format(l))
    # print("l_idx:\n{}".format(l_idx))
    # cycle_length = calc_cycle_length(l_idx, l)
    # print("cycle_length: {}".format(cycle_length))

    # print("t_last: {}".format(t_last))
    # print("i_first: {}".format(i_first))
    # print("i_last: {}".format(i_last))
    # print("cycle_length: {}".format(cycle_length))

    save_pkl_gz_obj(d_s_cycle_length, obj_d_s_cycle_length_file_path)
