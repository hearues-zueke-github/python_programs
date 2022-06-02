#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

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
import multiprocessing as mp

from collections import defaultdict
from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from hashlib import sha256
from io import BytesIO
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

class SingleLinearSequence(Exception):
    def __init__(self, m, a, b, x):
        assert (m & (m-1) == 0) and m != 0

        self.m = m

        a = a % m
        b = b % m

        self.a = a - a % 4 - 1
        self.b = b - b % 4 - 1
        self.x = x % m

    def next_num(self):
        self.x = ((self.a * self.x) + self.b) % self.m

        return self.x


class MultiLinearSequnce(Exception):
    def __init__(self, m, l_a, l_b, l_x, l_i=None, repeat_num=2):
        assert (m & (m-1) == 0) and m != 0

        self.m = m

        self.k = len(l_a)
        assert self.k == len(l_b)
        assert self.k == len(l_x)

        l_a = [a % m for a in l_a]
        l_b = [b % m for b in l_b]

        self.l_a = [a - a % 4 + 1 for a in l_a]
        self.l_b = [b - b % 2 + 1 for b in l_b]
        self.l_x = [x % m for x in l_x]

        if l_i is not None:
            self.l_i = [i % 2 for i in l_i]
        else:
            self.l_i = [0 for _ in range(0, len(l_a))]

        self.repeat_num = repeat_num

        self.acc = 0


    def next_num(self):
        calc_new_x = True
        v = 0
        for j in range(0, self.k):
            if calc_new_x:
                self.l_x[j] = ((self.l_a[j] * self.l_x[j]) + self.l_b[j]) % self.m
                self.l_i[j] += 1
                calc_new_x = False

            v = (v + self.l_x[j]) % self.m
            # v = (v ^ self.l_x[j]) % self.m
            
            # if j % 2 == 0:
            #     v = (v ^ self.l_x[j]) % self.m
            # else:
            #     v = (v + self.l_x[j]) % self.m

            if self.l_i[j] >= self.repeat_num:
                self.l_i[j] = 0

                if j < self.k - 1:
                    self.l_i[j + 1] += 1
                    calc_new_x = True

        self.acc = (self.acc ^ v) % self.m

        # return v
        return self.acc


if __name__ == '__main__':
    m = 2**2 # 4 bits

    # m = 2**3
    # array([[    8,     8,     8,     8,     8,     8],
    #    [   16,    12,    32,     5,    48,    28],
    #    [    8,    48,    64,   120,    72,   224],
    #    [   16,    96,   128,   180,   432,   896],
    #    [    8,   192,   256,  1080,   648,  3584],
    #    [   16,   384,   512,  1620,  3888, 14336],
    #    [    8,   768,  1024,  9720,  5832, 57344],
    #    [   16,  1536,  2048, 14580, 34992,    -1],
    #    [    8,  3072,  4096,    -1, 52488,    -1],
    #    [   16,  6144,  8192,    -1,    -1,    -1],
    #    [    8, 12288, 16384,    -1,    -1,    -1]])

    amount = 7
    l_a = [2, 3, 6, 10, 15, 3, 2, 7, 0, 3, 6][:amount]
    l_b = [1, 2, 8, 7, 1, 4, 5, 4, 2, 7, 2][:amount]
    l_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][:amount]
    
    arr_cycles = np.zeros((len(l_a), m-1), dtype=np.int64) - 1

    for amount_vals in range(1, len(l_a)+1):
        for repeat_num in range(2, m+1):
            print(f"amount_vals: {amount_vals}, repeat_num: {repeat_num}")
            mls = MultiLinearSequnce(
                m=m,
                l_a=l_a[:amount_vals],
                l_b=l_b[:amount_vals],
                l_x=l_x[:amount_vals],
                repeat_num=repeat_num,
            )

            l = []
            for j in range(0, 250000):
                v = mls.next_num()
                l.append(v)

            amount = len(l)
            max_third = (amount // 3) - 1

            found_nr = 0
            for cycle_length in range(1, max_third + 1):
                i1 = amount - cycle_length * 3
                i2 = amount - cycle_length * 2
                i3 = amount - cycle_length * 1
                i4 = amount - cycle_length * 0

                l1 = l[i1:i2]
                l2 = l[i2:i3]
                l3 = l[i3:i4]

                # if l1 == l2:
                #     print(f"found_nr: {found_nr}, cycle_length: {cycle_length}, l1 == l2")
                if l1 == l2 and l2 == l3:
                    if found_nr == 0:
                        if cycle_length == 1:
                            continue
                        arr_cycles[amount_vals-1, repeat_num-2] = cycle_length

                    print(f"- found_nr: {found_nr}, cycle_length: {cycle_length}, l1 == l2, l2 == l3")
                    found_nr += 1

                    if found_nr >= 2:
                        break

    # print(f"l: {l}")

    sys.exit()

    l_a = [6, 7, 3]
    l_b = [2, 7, 4]

    l_a = [b % m for a in l_a]
    l_b = [b % m for b in l_b]
    
    l_a = [a - a % 4 + 1 for a in l_a]
    l_b = [b - b % 2 + 1 for b in l_b]

    l_x = [0 for _ in l_a]
    l = [[l_x[i]] for i in range(0, len(l_a))]

    for _ in range(1, m):
        for i in range(0, len(l_a)):
            l_x[i] = (l_a[i] * l_x[i] + l_b[i]) % m
            l[i].append(l_x[i])

    print(f"l_a: {l_a}, l_b: {l_b}")

    for li in l:
        print(f"- li: {li}")
        u, c = np.unique(li, return_counts=True)
        print(f"-- u: {u}")
        print(f"-- c: {c}")

    assert np.all(c == 1)
