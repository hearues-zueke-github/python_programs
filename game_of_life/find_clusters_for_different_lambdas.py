#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import string
import sys
import inspect
import textwrap

from typing import List, Dict, Set, Mapping, Any, Tuple

# import tempfile
from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

from datetime import datetime
from collections import defaultdict
from copy import deepcopy
from dotmap import DotMap
from operator import itemgetter

from pprint import pprint

from os.path import expanduser

import itertools

import multiprocessing as mp

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
import global_object_getter_setter
import utils

sys.path.append('../clustering')
import utils_cluster

from bit_automaton import BitAutomaton

if __name__ == '__main__':
    dir_path_save_images = os.path.join(TEMP_DIR, 'save_images/')
    assert os.path.exists(dir_path_save_images)

    dm_obj_file_name = utils_cluster.dm_obj_file_name

    frame = 2

    l_arr_historic_ranges : List[np.ndarray] = []
    l_func_str : List[str] = []
    amount : int = 0
    for root, dirs, files in os.walk(dir_path_save_images):
        if dm_obj_file_name in files:
            with gzip.open(os.path.join(root, dm_obj_file_name), 'rb') as f:
                dm_obj = dill.load(f)
            if dm_obj.frame == frame:
                amount += 1
                print("amount: {}".format(amount))
                l_arr_historic_ranges.append(dm_obj.arr_historic_ranges)
                l_func_str.append(dm_obj.func_str)

    l_arr_nr = [i for i, arr in enumerate(l_arr_historic_ranges, 0) for _ in range(0, arr.shape[0])]
    arr_historic_ranges_all = np.vstack(l_arr_historic_ranges)

    cluster_points, arr_error = utils_cluster.calculate_clusters(
        points=arr_historic_ranges_all.astype(np.float),
        cluster_amount=4,
        iterations=100,
    )
