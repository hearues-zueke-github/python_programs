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
    dir_path = "/tmp/generate_2d_sequence"

    d_params = dict(
        modulo=6,
        const=5,
        a=4,
        b=0,
        c=0,
    )

    file_name_cpp_base = f"m_{d_params['modulo']:02}_v_a_{d_params['a']:02}_v_b_{d_params['b']:02}_v_c_{d_params['c']:02}_v_const_{d_params['const']:02}"
    file_name_cpp_data = f"{file_name_cpp_base}.data"
    file_path_cpp_data = os.path.join(dir_path, file_name_cpp_data)

    print(f"file_path_cpp_data: {file_path_cpp_data}")

    with open(file_path_cpp_data, "rb") as f:
        content = f.read()

    idx = 0

    jmp = 8 * 5
    arr_0 = np.frombuffer(buffer=content[idx:idx+jmp], dtype=np.uint64)
    idx += jmp

    modulo = arr_0[0]
    v_const = arr_0[1]
    v_a = arr_0[2]
    v_b = arr_0[3]
    v_c = arr_0[4]

    jmp = 1
    arr_1 = np.frombuffer(buffer=content[idx:idx+jmp], dtype=np.uint8)
    idx += jmp

    dim = arr_1[0]
    assert dim == 2
    
    jmp = 8 * 3
    arr_2 = np.frombuffer(buffer=content[idx:idx+jmp], dtype=np.uint64)
    idx += jmp

    rows = arr_2[0]
    cols = arr_2[1]
    amount = arr_2[2]

    assert rows * cols == amount

    dtype = np.uint8
    bytes_type = dtype(0).nbytes
    jmp = bytes_type * amount.astype(np.int64)
    arr = np.frombuffer(buffer=content[idx:idx+jmp], dtype=dtype).reshape((rows, cols))
    idx += jmp

    d_cpp = dict(
        modulo=modulo,
        v_const=v_const,
        v_a=v_a,
        v_b=v_b,
        v_c=v_c,
        rows=rows,
        cols=cols,
        amount=amount,
        arr=arr,
    )

    print(f"d_cpp: {d_cpp}")

    file_name_py_base = f"m_{d_params['modulo']:02}_fac_const_{d_params['const']:02}_fac_a_{d_params['a']:02}_fac_b_{d_params['b']:02}_fac_c_{d_params['c']:02}"
    file_name_pkl = f"{file_name_py_base}.pkl"
    file_path_pkl = os.path.join(dir_path, file_name_pkl)

    with open(file_path_pkl, 'rb') as f:
        d_py = dill.load(f)

    arr_cpp = d_cpp['arr']
    arr_py = d_py['arr']

    shape_cpp = arr_cpp.shape
    shape_py = arr_py.shape

    min_y = min(shape_cpp[0], shape_py[0])
    min_x = min(shape_cpp[1], shape_py[1])

    arr_cpp_part = arr_cpp[:min_y, :min_x]
    arr_py_part = arr_py[:min_y, :min_x]

    assert np.all(arr_cpp_part == arr_py_part)

    file_name_py_png = f"{file_name_py_base}.png"
    file_name_cpp_png = f"{file_name_cpp_base}.png"

    img_py = Image.open(os.path.join(dir_path, file_name_py_png))
    img_cpp = Image.open(os.path.join(dir_path, file_name_cpp_png))

    pix_py = np.array(img_py)
    pix_cpp = np.array(img_cpp)

    assert np.all(pix_py[:min_y, :min_x]==pix_cpp[:min_y, :min_x])
