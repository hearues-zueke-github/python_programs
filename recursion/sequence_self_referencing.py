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

if __name__ == '__main__':
    n_max = 20000
    d = {}
    for jump in range(1, 51):
        print()

        l = [0, 0]

        for i in range(2, n_max):
            i1 = l[i - 1]
            i2 = l[i - 2]

            x1 = l[i1]
            x2 = l[i2]

            j1 = (x1 + x2) % i
            j2 = (x1 * x2) % i

            xj_1 = l[j1]
            xj_2 = l[j2]

            # x = (x1 + x2 + jump) % i
            x = (xj_1 + xj_2 + jump) % i

            l.append(x)

            if i > 100:
                is_cycling = False
                for j in range(1, i//3):
                    if l[-j:] == l[-2*j:-j] and l[-1*j:] == l[-3*j:-2*j]:
                        is_cycling = True
                        break
                if is_cycling:
                    print("Is cycling!!")
                    break

        # print(f"jump: {jump}, l: {l}")
        d[jump] = l
        print(f"jump: {jump}")
        u, c = np.unique(l, return_counts=True)
        print(f"- u.shape: {u.shape}")
        # print(f"- u: {u.tolist()}")
        # print(f"- c: {c.tolist()}")

