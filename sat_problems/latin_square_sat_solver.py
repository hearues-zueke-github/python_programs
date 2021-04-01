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

import multiprocessing as mp
import numpy as np
import pandas as pd
from z3 import *

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from pprint import pprint
from shutil import copyfile

sys.path.append('..')
from utils import mkdirs
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

from pysat.solvers import Glucose3

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':
    print('Hello World!')

    n = 10
    arr_cells = np.array([IntVector('cell_{}'.format(row), n) for row in range(0, n)])
    
    l_cells_trans = list(zip(*arr_cells))

    s = Solver()

    for arr_row in arr_cells:
        for var in arr_row:
            s.add(var >= 0, var < n)

    for arr_row in arr_cells:
        for i in range(0, n):
            s.add(Sum([If(v == i, 1, 0) for v in arr_row]) == 1)

    for l_col in l_cells_trans:
        for i in range(0, n):
            s.add(Sum([If(v == i, 1, 0) for v in l_col]) == 1)

    # max_jumps and min possible n: [(0, 1), (1, 2), (2, 5), (3, 10), (4, )]
    max_jumps = 2
    # add first neighbor diagonals!
    for jump in range(1, max_jumps+1):
        for j in range(0, n - jump):
            for i in range(0, n - jump):
                s.add(arr_cells[j, i] != arr_cells[j + jump, i + jump])
                s.add(arr_cells[j, i + jump] != arr_cells[j + jump, i])

        for jump_x in range(1, jump):
            for j in range(0, n - jump):
                for i in range(0, n - jump_x):
                    s.add(arr_cells[j, i] != arr_cells[j + jump, i + jump_x])
                    s.add(arr_cells[j, i + jump_x] != arr_cells[j + jump, i])

        for jump_y in range(1, jump):
            for j in range(0, n - jump_y):
                for i in range(0, n - jump):
                    s.add(arr_cells[j, i] != arr_cells[j + jump_y, i + jump])
                    s.add(arr_cells[j, i + jump] != arr_cells[j + jump_y, i])


    # add constraint for having a specific color combo for edges and corners
    # also add a constraint that only one unique edge and one unique corner do exist!

    print('Starting the check for the solution!')
    print(s.check())
    # print(s.model())

    m = s.model()

    arr_cells_val = np.array([[(lambda d: int(d.sexpr()))(m[v]) for v in arr_row] for arr_row in arr_cells])
    print("arr_cells_val:\n{}".format(arr_cells_val))
    # arr_cells_val_trans = arr_cells_val.T
    # print("arr_cells_val_trans:\n{}".format(arr_cells_val_trans))
