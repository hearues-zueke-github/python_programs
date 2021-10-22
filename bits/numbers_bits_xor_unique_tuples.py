#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9.5 -i

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
from typing import List, Set, Tuple, Dict, Union
from PIL import Image

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append('..')
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PATH_ROOT_DIR, "../utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PATH_ROOT_DIR, "../utils_multiprocessing_manager.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(PLOTS_DIR_PATH)

if __name__ == '__main__':
    # print("Hello World!")
    d_len_s = {}
    for bits in range(1, 9):
        # bits = 2
        n = 2**bits
        l = []
        for x in range(0, n):
            for y in range(0, n):
                l.append([x, y, x^y])

        # print("l: {}".format(l))

        s = set([tuple(sorted(t)) for t in l])
        # print("s: {}".format(s))
        print("len(s): {}".format(len(s)))

        d_len_s[bits] = len(s)

    print("d_len_s: {}".format(d_len_s))
