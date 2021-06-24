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

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':
    print("Hello World!")
