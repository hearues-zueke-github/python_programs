#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9.5 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import io
import os
import pdb
import re
import sys
import tarfile
import traceback

import numpy as np
import pandas as pd
import multiprocessing as mp

from collections import defaultdict
from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
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
    file_bytes_dill = io.BytesIO()
    file_bytes_dill2 = io.BytesIO()
    file_bytes_tar = io.BytesIO()
    file_tar_file = tarfile.open(fileobj=file_bytes_tar, mode="w:gz")
    # file_bytes_gzip = io.BytesIO()
    # file_tar_file = tarfile.TarFile(fileobj=file_bytes_tar, mode="w:gz")
    # file_gzip_file = gzip.GzipFile(fileobj=file_bytes_gzip, mode="w")

    d = {
        'foo': 'bar',
        'test': 1234,
        314: ['not', 'pi'],
    }

    d2 = {
        'foo2': 'bar',
        'test1': 1234,
        628: ['not', '2times', 'pi'],
    }

    dill.dump(d, file_bytes_dill)
    dill.dump(d2, file_bytes_dill2)

    tarinfo = tarfile.TarInfo(name='fold/ers/d.pkl')
    tarinfo.size = file_bytes_dill.tell()
    file_bytes_dill.seek(0)
    file_tar_file.addfile(tarinfo=tarinfo, fileobj=file_bytes_dill)

    tarinfo = tarfile.TarInfo(name='folder2/d2.pkl')
    tarinfo.size = file_bytes_dill2.tell()
    file_bytes_dill2.seek(0)
    file_tar_file.addfile(tarinfo=tarinfo, fileobj=file_bytes_dill2)

    file_tar_file.close()

    # file_bytes_tar.seek(0)
    # file_gzip_file.write(file_bytes_tar.read())

    file_bytes_dill.seek(0)
    file_bytes_dill2.seek(0)
    file_bytes_tar.seek(0)
    # file_bytes_gzip.seek(0)

    with open(os.path.join(TEMP_DIR, 'd.pkl'), 'wb') as f:
        f.write(file_bytes_dill.read())
    with open(os.path.join(TEMP_DIR, 'd2.pkl'), 'wb') as f:
        f.write(file_bytes_dill2.read())
    with open(os.path.join(TEMP_DIR, 'objs.tar.gz'), 'wb') as f:
        f.write(file_bytes_tar.read())
    # with open(os.path.join(TEMP_DIR, 'objs.gz'), 'wb') as f:
    #     f.write(file_bytes_gzip.read())
