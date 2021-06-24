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

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs'
mkdirs(OBJS_DIR_PATH)

import decimal
decimal.getcontext().prec = 130

Dec = decimal.Decimal

def get_best_fitting_value_for_the_equation(k: int, iterations: int, a_init: decimal.Decimal=None) -> decimal.Decimal:
    dec_0 = Dec(0)
    dec_1 = Dec(1)
    dec_delta = Dec(1) / Dec(3000)

    k = Dec(k)
    if a_init is None:
        a = k + 1
    else:
        assert isinstance(a_init, decimal.Decimal)

    for i in range(1, iterations+1):
        x = (a - k)**a - a**(a - k)
        a = a - dec_delta * x

    print("k: {}, a: {}".format(k, a))

    return (k, iterations), a


def calc_diff_of_values(l: List[decimal.Decimal]) -> List[decimal.Decimal]:
    return [v2 - v1 for v1, v2 in zip(l[:-1], l[1:])]


if __name__ == '__main__':
    iterations = 13000
    l_k = [i for i in range(1, 61)]

    mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())
    f = get_best_fitting_value_for_the_equation

    print('Define new Function!')
    mult_proc_mng.define_new_func('func_f', f)
    print('Do the jobs!!')
    l_arguments = [(k, iterations) for k in l_k]
    l_ret = mult_proc_mng.do_new_jobs(
        ['func_f']*len(l_arguments),
        l_arguments,
    )
    print("len(l_ret): {}".format(len(l_ret)))

    # testing the responsivness again!
    mult_proc_mng.test_worker_threads_response()
    del mult_proc_mng

    l_ret_filtered = list(filter(lambda x: x is not None, l_ret))

    l_ret_sorted = sorted(l_ret)
    l_a = [a for _, a in l_ret_sorted]
    l_a_diff = calc_diff_of_values(l=l_a)
    l_a_diff_diff = calc_diff_of_values(l=l_a_diff)
    l_a_diff_diff_diff = calc_diff_of_values(l=l_a_diff_diff)
