#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9.5
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

import importlib.util as imp_util

spec = imp_util.spec_from_file_location("utils", os.path.join(PATH_ROOT_DIR, "../utils.py"))
utils = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils)

mkdirs = utils.mkdirs

spec = imp_util.spec_from_file_location("utils_multiprocessing_manager", os.path.join(PATH_ROOT_DIR, "../utils_multiprocessing_manager.py"))
utils_multiprocessing_manager = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils_multiprocessing_manager)

MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

spec = imp_util.spec_from_file_location("utils_serialization", os.path.join(PATH_ROOT_DIR, "../utils_serialization.py"))
utils_serialization = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils_serialization)

get_pkl_gz_obj = utils_serialization.get_pkl_gz_obj
save_pkl_gz_obj = utils_serialization.save_pkl_gz_obj
load_pkl_gz_obj = utils_serialization.load_pkl_gz_obj

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

def get_d_l_t_2d_cells():
    def load_d_l_t_2d_cells():
        d_l_t_2d_cells = {}
        for height in range(1, 5):
            for width in range(1, 5):
                print("height: {}, width: {}".format(height, width))

                s_t_2d_cells = set()

                l_l_bit = []
                l_arr = []
                for v in range(1, 2**(height * width)):
                    l_bit = [int(i) for i in bin(v)[2:].zfill(height * width)]
                    l_l_bit.append(l_bit)
                    arr = np.array(l_bit).reshape((height, width))
                    l_arr.append(arr)

                    arr_y, arr_x = np.where(arr)
                    arr_y -= np.min(arr_y)
                    arr_x -= np.min(arr_x)

                    t = tuple(sorted([(y, x) for y, x in zip(arr_y, arr_x)]))
                    if t not in s_t_2d_cells:
                        s_t_2d_cells.add(t)

                print("len(l_l_bit): {}".format(len(l_l_bit)))
                print("len(s_t_2d_cells): {}".format(len(s_t_2d_cells)))

                l_t_2d_cells = [t for _, t in sorted([(len(t), t) for t in s_t_2d_cells])]

                d_l_t_2d_cells[(height, width)] = l_t_2d_cells

        return d_l_t_2d_cells

    obj = get_pkl_gz_obj(load_d_l_t_2d_cells, os.path.join(TEMP_DIR, 'd_l_t_2d_cells.pkl.gz'))
    return obj

if __name__ == '__main__':
    d_l_t_2d_cells = get_d_l_t_2d_cells()
    print("d_l_t_2d_cells.keys(): {}".format(d_l_t_2d_cells.keys()))
