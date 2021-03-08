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

import numpy as np
import pandas as pd

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile

import config_file

sys.path.append('..')
from utils import mkdirs
from utils_multiprocessing_manager import MultiprocessingManager
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

sys.path.append('../combinatorics')
import different_combinations

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':    
    def do_simple_loop(n):
        for _ in range(0, n):
            pass
        return None

    mult_proc_manag = MultiprocessingManager(cpu_count=4)
    mult_proc_manag.define_new_func(name='do_simple_loop', func=do_simple_loop)
    
    l_func_args = config_file.l_func_args
    l_func_name = ['do_simple_loop'] * len(l_func_args)
    mult_proc_manag.do_new_jobs(l_func_name=l_func_name, l_func_args=l_func_args)

    del mult_proc_manag
