#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import hashlib
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
sys.path.append(os.path.join(HOME_DIR, 'git/python_programs'))
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

BLOCKSIZE = 65536
def calc_sha256sum_from_file_path(file_path):
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        file_buffer = f.read(BLOCKSIZE)
        while len(file_buffer) > 0:
            h.update(file_buffer)
            file_buffer = f.read(BLOCKSIZE)
        
    return h.hexdigest()


if __name__ == '__main__':
    print("CURRENT_WORKING_DIR: {}".format(CURRENT_WORKING_DIR))

    root_first = CURRENT_WORKING_DIR
    l_file_path = []
    for root, l_dir_name, l_file_name in os.walk(root_first):
        for file_name in l_file_name:
            l_file_path.append(os.path.join(root, file_name))
    pprint(l_file_path)

    l_file_path_reduced = [file_path.replace(root_first, '')[1:] for file_path in l_file_path]
    l_hashdigest = [calc_sha256sum_from_file_path(file_path) for file_path in l_file_path]

    df = pd.DataFrame(
        data={
            'l_file_path': l_file_path,
            'l_file_path_reduced': l_file_path_reduced,
            'l_hashdigest': l_hashdigest,
        },
        columns=[
            'l_file_path',
            'l_file_path_reduced',
            'l_hashdigest',
        ],
        dtype=object,
    )
