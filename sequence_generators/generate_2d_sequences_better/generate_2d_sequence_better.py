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
    print("Hello World!")

    def get_f(m, fac_const, fac_a, fac_b, fac_c):
        def f(a, b, c):
            return (a * fac_a + b * fac_b + c * fac_c + fac_const) % m
        return f

    # n = 1024
    # n = 128
    n = 256

    # fac_const = 1
    # assert fac_const >= 0 and fac_const < m

    dir_path_temp = os.path.join(TEMP_DIR, f'generate_2d_sequence')
    # dir_path_temp = os.path.join(TEMP_DIR, f'generate_2d_sequence/m_{m:02}_fac_const_{fac_const:02}')
    mkdirs(dir_path_temp)

    def f(m, fac_const, fac_a, fac_b, fac_c):
        print(f"m: {m}, fac_const: {fac_const}, fac_a: {fac_a}, fac_b: {fac_b}, fac_c: {fac_c}")
        arr = np.zeros((n, n), dtype=np.uint8)

        # arr[0, :] = np.tile(np.arange(0, 2), n // 2)
        # arr[:, 0] = np.tile(np.arange(0, 2), n // 2)

        arr[0, :] = np.arange(0, n) % m
        arr[:, 0] = np.arange(0, n) % m

        f = get_f(m=m, fac_const=fac_const, fac_a=fac_a, fac_b=fac_b, fac_c=fac_c)
        for y in range(1, n):
            for x in range(1, n):
                arr[y, x] = f(a=arr[y-1, x], b=arr[y, x-1], c=arr[y-1, x-1])

        arr_idx_to_val = (np.arange(0, 1.0000000000001, 1 / (m-1)) * 256).astype(np.uint8)
        arr_idx_to_val[-1] = 255
        assert arr_idx_to_val.shape[0] == m

        img = Image.fromarray(arr_idx_to_val[arr])

        # factor_resize = 2
        # size = img.size
        # img = img.resize(size=(size[0]*factor_resize, size[1]*factor_resize), resample=Image.NONE)

        base_file_name = f'm_{m:02}_fac_const_{fac_const:02}_fac_a_{fac_a:02}_fac_b_{fac_b:02}_fac_c_{fac_c:02}'
        img.save(os.path.join(dir_path_temp, f"{base_file_name}.png"))

        file_path_pkl = os.path.join(dir_path_temp, f"{base_file_name}.pkl")
        with open(file_path_pkl, 'wb') as f:
            dill.dump(dict(
                m=m,
                v_const=fac_const,
                v_a=fac_a,
                v_b=fac_b,
                v_c=fac_c,
                arr=arr,
            ), f)

        return ((m, fac_const, fac_a, fac_b, fac_c), arr)

    mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())

    # # only for testing the responsivness!
    # mult_proc_mng.test_worker_threads_response()

    print('Define new Function!')
    mult_proc_mng.define_new_func('func_f', f)
    print('Do the jobs!!')
    l_arguments = []
    for _ in range(0, 1):
        m = np.random.randint(5, 10, (1, ))[0]
        l_arguments.append((m, )+tuple(np.random.randint(0, m, (4, )).tolist()))
    l_arguments = sorted(set(l_arguments))
    # for fac_c in range(0, m):
    #     for fac_a in range(0, m):
    #         for fac_b in range(0, m):
    #             l_arguments.append((fac_a, fac_b, fac_c))
    l_ret = mult_proc_mng.do_new_jobs(
        ['func_f']*len(l_arguments),
        l_arguments,
    )
    print("len(l_ret): {}".format(len(l_ret)))
    # print("l_ret: {}".format(l_ret))

    # # testing the responsivness again!
    # mult_proc_mng.test_worker_threads_response()
    del mult_proc_mng

    print(f"l_ret[0]:\n{l_ret[0]}")
