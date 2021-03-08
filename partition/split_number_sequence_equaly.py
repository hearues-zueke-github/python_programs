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

from typing import List

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from pprint import pprint
from shutil import copyfile

sys.path.append('..')
from utils import mkdirs

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':
    print("Hello World!")

    l = np.array([1, 2, 3, 3, 3, 4, 4, 5, 6, 6])
    # l = [1, 2, 3, 3, 3, 4, 4, 5, 6, 6]
    partitions = 3

    def calc_random_partion_sum_len_diff_sum(l, partitions):
        arr_idxs_nr = np.arange(0, len(l))
        arr_idxs_nr_perm = np.random.permutation(arr_idxs_nr)

        l_parts_arr = [np.sort(l[arr_idxs_nr_perm[i::partitions]]) for i in range(0, partitions)]
        # print("l_parts: {}".format(l_parts))
        l_parts = sorted(list(map(list, l_parts_arr)))

        l_sum = [sum(a) for a in l_parts]
        l_len = [len(a) for a in l_parts]

        # print("l_sum: {}".format(l_sum))
        # print("l_len: {}".format(l_len))

        def calc_absolute_diff_sum(l_sum: List[List[int]]) -> int:
            s = 0
            
            for i1, v1 in enumerate(l_sum[:-1], 0):
                for i2, v2 in enumerate(l_sum[i1+1:], i1+1):
                    s += abs(v1 - v2)

            return s

        absolute_diff_sum = calc_absolute_diff_sum(l_sum)

        return tuple(map(tuple, l_parts)), tuple(l_sum), tuple(l_len), absolute_diff_sum

    s_tpl = set()
    for i in range(0, 100):
        print("i: {}".format(i))
        tpl = calc_random_partion_sum_len_diff_sum(l=l, partitions=partitions)
        l_parts, l_sum, l_len, absolute_diff_sum = tpl
        tpl2 = (absolute_diff_sum, l_sum, l_len, l_parts)
        
        if tpl2 in s_tpl:
            print("- Ignore this one!")
            continue
        
        s_tpl.add(tpl2)

    l_tpl = sorted(s_tpl)
    pprint(l_tpl)
