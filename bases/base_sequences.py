#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9 -i

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

import matplotlib.pyplot as plt

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

from multiprocessing.managers import SharedMemoryManager

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
load_module_dynamically(**dict(var_glob=var_glob, name='different_combinations', path=os.path.join(PYTHON_PROGRAMS_DIR, "combinatorics/different_combinations.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_graph_theory', path=os.path.join(PYTHON_PROGRAMS_DIR, "graph_theory/utils_graph_theory.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager
get_all_combinations_repeat = different_combinations.get_all_combinations_repeat
get_cycles_of_1_directed_graph = utils_graph_theory.get_cycles_of_1_directed_graph

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

def convert_num_to_base_num(n, b, min_len=-1):
    def gen(n):
        while n > 0:
            yield n%b; n //= b
    l = [i for i in gen(n)]
    return l if min_len == -1 else l+[0]*(min_len - len(l) if min_len > len(l) else 0)

def get_num_from_base_lst(l, b):
    n = 0
    mult = 1
    for i, v in enumerate(l, 0):
        n += v*mult
        mult *= b
    return n


def calc_iters_from_base_to_base(b1, b2):
    # b1 = 3
    # b2 = 7

    assert b1 > 1
    assert b2 > 1
    assert b1 < b2

    l_a = []
    # print(f"b1: {b1}, b2: {b2}")
    for n in range(1, 40+1):
        num = b1**n
        iters = 0
        while num >= b1:
            l = convert_num_to_base_num(n=num, b=b2)
            l2 = [i % b1 for i in l]
            num = get_num_from_base_lst(l=l2, b=b1)
            iters += 1

        # print(f"n: {n}, iters: {iters}")
        l_a.append(iters)

    # print(f"l_a: {l_a}")
    return l_a


if __name__ == '__main__':
    b1 = 3
    b2 = 7

    assert b1 > 1
    assert b2 > 1
    assert b1 < b2

    l_a = []
    # print(f"b1: {b1}, b2: {b2}")
    for n in [5]:
    # for n in range(1, 40+1):
        num = b1**n
        iters = 0
        l_num = []
        while num >= b1:
            l = convert_num_to_base_num(n=num, b=b2)
            l2 = [i % b1 for i in l]
            num = get_num_from_base_lst(l=l2, b=b1)
            iters += 1
            l_num.append(num)

        # print(f"n: {n}, iters: {iters}")
        l_a.append(iters)

    # print(f"l_a: {l_a}")
    print(f"b1: {b1}")
    print(f"b2: {b2}")
    print(f"l_num: {l_num}")
    # return l_a
