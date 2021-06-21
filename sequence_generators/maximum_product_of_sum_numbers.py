#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

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

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()

import importlib.util as imp_util

# TODO: change the optaining of the git root folder from a config file first!
spec = imp_util.spec_from_file_location("utils", os.path.join(HOME_DIR, "git/python_programs/utils.py"))
utils = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils)

mkdirs = utils.mkdirs

spec = imp_util.spec_from_file_location("utils_multiprocessing_manager", os.path.join(HOME_DIR, "git/python_programs/utils_multiprocessing_manager.py"))
utils_multiprocessing_manager = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils_multiprocessing_manager)

MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

spec = imp_util.spec_from_file_location("prime_numbers_fun", os.path.join(HOME_DIR, "git/python_programs/math_numbers/prime_numbers_fun.py"))
prime_numbers_fun = imp_util.module_from_spec(spec)
spec.loader.exec_module(prime_numbers_fun)

get_primes = prime_numbers_fun.get_primes

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs'
mkdirs(OBJS_DIR_PATH)

class DictSum(Exception):
    def __init__(self, d_values: Dict[Any, int]={}):
        self.d_values: Dict[Any, int] = d_values

    def add(self, other: 'DictSum') -> None:
        d_1 = self.d_values
        d_2 = other.d_values

        s_keys_1 = set(d_1.keys())
        s_keys_2 = set(d_2.keys())

        for k in s_keys_1 & s_keys_2:
            d_1[k] += d_2[k]

        for k in s_keys_2 - s_keys_1:
            d_1[k] = d_2[k]

    def __repr__(self):
        return 'DictSum({})'.format(str(self.d_values))

    def __str__(self):
        return 'DictSum({})'.format(str(self.d_values))

    def tuple(self):
        return tuple(sorted([(k, v) for k, v in self.d_values.items()]))


if __name__ == '__main__':
    print("Hello World!")

    n_max = 70
    l_p = list(get_primes(n_max*2))

    # n = 14
    # get all possible sum arrays!

    # dsum_1 = DictSum({2:1, 3:2})
    # dsum_2 = DictSum({3:1, 5:7})

    # d_n_splits = {
    #     2: [DictSum({2: 1})],
    #     3: [DictSum({3: 1})],
    # }

    def get_all_prime_addition_list_splits(n: int) -> List[List[int]]:
        l_l_splits: List[List[int]] = []

        max_n_half = n // 2
        l_num = [l_p[0]]
        l_num_idx = [0]
        sum_l = l_num[0]
        # first_idx = 0
        while l_num[0] <= max_n_half:
            if sum_l < n:
                p_idx = l_num_idx[-1]
                p = l_p[p_idx]
                l_num.append(p)
                l_num_idx.append(p_idx)
                sum_l += p
            elif sum_l >= n:
                if sum_l == n:
                    l_l_splits.append(list(l_num))
                    # print("l_l_splits: {}".format(l_l_splits))

                    # print("l_num: {}".format(l_num))
                    # print("l_num_idx: {}".format(l_num_idx))
                    # input('\nENTER...')

                p_prev_del = l_num.pop()
                p_idx_prev = l_num_idx.pop()
                l_num_idx[-1] += 1

                p = l_p[l_num_idx[-1]]
                p_prev = l_num[-1]
                l_num[-1] = p

                # if len(l_num) == 1:
                #     sum_l = sum_l - p_prev_del
                # else:
                sum_l = sum_l - p_prev_del - p_prev + p

                # assert False

            # print("l_num: {}".format(l_num))
            # print("l_num_idx: {}".format(l_num_idx))
            # print()
            # input()
            
            # print("l_num: {}".format(l_num))
            # print("l_num_idx: {}".format(l_num_idx))
            # print("sum_l: {}".format(sum_l))
            # input('\nENTER...')

        return l_l_splits

    d_n_max_prod = {}
    d_n_splits = {}

    # n = 10
    for n in range(4, n_max+1):
        print("n: {}".format(n))

        l_l_splits = get_all_prime_addition_list_splits(n=n)
        assert all([sum(l)==n for l in l_l_splits])
        s = set([tuple(l) for l in l_l_splits])
        assert len(s) == len(l_l_splits)
        l_prod = [np.prod(l) for l in l_l_splits]
        max_prod_idx = np.argmax(l_prod)
        max_prod = l_prod[max_prod_idx]

        d_n_max_prod[n] = dict(max_prod=max_prod, l=l_l_splits[max_prod_idx])
        d_n_splits[n] = l_l_splits

    l_tpl = [(2, 2), (3, 3)] + [(lambda d: (i, d['max_prod']))(d_n_max_prod[i]) for i in range(4, n_max+1)]

    # potential for a new entry into the oeis.org site
    l_length_addition_splits = [len(d_n_splits[i]) for i in range(4, n_max)]

    sys.exit()

    for n in range(4, n_max+1):
        l_splits = [DictSum({n: 1})]

        # add all simple splits to the number possible, starting from 2+(n-2), 3+(n-3) etc.
        for i in range(2, n//2+1):
            dsum = DictSum({i: 1})
            dsum.add(DictSum({n-i: 1}))

