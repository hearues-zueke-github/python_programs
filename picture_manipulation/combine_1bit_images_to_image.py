#! /usr/bin/env -S /usr/bin/time /usr/bin/python3 -i

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys
import string

import shutil

from typing import List, Dict, Set, Mapping, Any, Tuple

# import tempfile
from memory_tempfile import MemoryTempfile
tempfile : MemoryTempfile = MemoryTempfile()

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
CURRENT_DIR = os.getcwdb().decode('utf-8')

from PIL import Image

import numpy as np
import pandas as pd

sys.path.append('..')
import utils
from utils_multiprocessing_manager import MultiprocessingManager


def convert_1d_to_2d_arr(arr, length):
    arr_2d = np.zeros((arr.shape[0]-length+1, length), dtype=np.uint8)
    for i in range(0, length-1):
        arr_2d[:, i] = arr[i:-length+1+i]
    arr_2d[:, -1] = arr[length-1:]
    return arr_2d


lst_int_base_100 = string.printable
# lst_int_base_100 = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_.,:;!?#$%&()[]{}/\\ \'\"")
base_100_len = len(lst_int_base_100)
assert base_100_len == 100
dict_base_100_int = {v: i for i, v in enumerate(lst_int_base_100, 0)}

def convert_base_100_to_int(num_base_100):
    b = 1
    s = 0
    for i, v in enumerate(reversed(list(num_base_100)), 0):
        n = dict_base_100_int[v]
        s += n*b
        b *= base_100_len
    return s


def convert_int_to_base_100(num_int):
    l = []
    while num_int > 0:
        l.append(num_int % base_100_len)
        num_int //= base_100_len
    n = list(map(lambda x: lst_int_base_100[x], reversed(l)))
    return "".join(n)


def convert_int_to_lst_bin(num_int):
    return list(map(int, bin(num_int)[2:]))


def convert_lst_bin_to_int(l_bin):
    arr = np.array(l_bin, dtype=object)
    length = arr.shape[0]
    return np.sum(arr*2**np.arange(length-1, -1, -1).astype(object))


secret_test = "test123%$&/?!-_,:.;"

assert secret_test==convert_int_to_base_100(convert_base_100_to_int(secret_test))
assert 12345678901234567890==convert_base_100_to_int(convert_int_to_base_100(12345678901234567890))

assert 1234567==convert_lst_bin_to_int(convert_int_to_lst_bin(1234567))
assert [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]==convert_int_to_lst_bin(convert_lst_bin_to_int([1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]))


arr_prefix : np.ndarray = np.array(convert_int_to_lst_bin(0xabcd), dtype=np.uint8)
arr_suffix : np.ndarray = np.array(convert_int_to_lst_bin(0x34bf), dtype=np.uint8)

len_arr_prefix = arr_prefix.shape[0]
len_arr_suffix = arr_suffix.shape[0]

if __name__ == '__main__':
    print('Hello World!')

    path_images = os.path.join(PATH_ROOT_DIR, 'images/')
    assert os.path.exists(path_images)

    img_src_path : str = "images/orig_image_2_no_secret.png"
    img_src_new_path : str = "images/orig_image_2_no_secret_new.png"
    img_dst_path : str = "images/orig_image_2_with_secret.png"
