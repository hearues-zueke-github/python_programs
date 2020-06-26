#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import re
import string
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats as st
# import scipy as sp

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"


def func_timer(f, args):
    start_time = time.time()
    r = f(*args)
    end_time = time.time()
    return r, end_time - start_time


if __name__ == '__main__':
    # create random arr nums with a dimension m and amount n
    dimension = 3
    n = 10000

    assert 256**dimension >= n

    arr_nums = np.random.randint(0, 2**8, (n, dimension))
    arr_nums_unique = np.unique(arr_nums.astype(np.uint8).reshape((-1, )).view(','.join(['u1']*dimension)))

    arr_nums_view = np.empty((len(arr_nums_unique), ), dtype=object)
    arr_nums_view[:] = arr_nums_unique


    arr_vals = np.random.randint(0, 2**32, (len(arr_nums_unique), ))
    
    arr_nums_uint8 = np.empty((len(arr_nums_unique), ), dtype=object)
    arr_nums_uint8[:] = list(map(tuple, arr_nums_unique))
    df_uint8 = pd.DataFrame(data={'val': arr_vals, 'nums': arr_nums_uint8}, columns=['nums', 'val'])
    # l_nums_uint8 = list(map(tuple, arr_nums_unique))
    # df_uint8 = pd.DataFrame(data={'val': arr_vals, 'nums': l_nums_uint8}, columns=['nums', 'val'])
    df_uint8_nums = df_uint8.set_index('nums')

    arr_nums_int = np.empty((len(arr_nums_unique), ), dtype=object)
    arr_nums_int[:] = list(map(lambda x: tuple(map(int, x)), arr_nums_unique))
    df_int = pd.DataFrame(data={'val': arr_vals, 'nums': arr_nums_int}, columns=['nums', 'val'])
    # l_nums_int = list(map(lambda x: tuple(map(int, x)), arr_nums_unique))
    # df_int = pd.DataFrame(data={'val': arr_vals, 'nums': l_nums_int}, columns=['nums', 'val'])
    df_int_nums = df_int.set_index('nums')

    def mapping_1(arr_nums, arr_vals):
        d = 4


    # def convert_num_to_8bit_array(num, dimension):
    #     l = []
    #     while num > 0:
    #         l.append(num % 256)
    #         num //= 256
    #     return l
    # arr_vals = [convert_num_to_8bit_array(num, dimension) for num in l_vals]
