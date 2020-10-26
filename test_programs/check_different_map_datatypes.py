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
    return end_time - start_time, r


if __name__ == '__main__':
    # create random arr nums with a dimension m and amount n
    dimension = 10
    n = 100000

    assert 256**dimension >= n

    arr_nums = np.random.randint(0, 2**8, (n, dimension))
    arr_nums_unique = np.unique(arr_nums.astype(np.uint8).reshape((-1, )).view(','.join(['u1']*dimension)))

    length = len(arr_nums_unique)
    arr_vals = np.random.randint(0, 2**32, (length, ))

    # df_uint8_view = pd.DataFrame(data={'val': arr_vals, 'nums': arr_nums_unique}, columns=['nums', 'val'])
    # df_uint8_view_nums = df_uint8_view.set_index('nums')    

    arr_nums_uint8 = np.empty((length, ), dtype=object)
    arr_nums_uint8[:] = list(map(tuple, arr_nums_unique))
    # df_uint8 = pd.DataFrame(data={'val': arr_vals, 'nums': arr_nums_uint8}, columns=['nums', 'val'])
    # df_uint8_nums = df_uint8.set_index('nums')
    df_uint8_nums = pd.Series(data=arr_vals, index=arr_nums_uint8)

    arr_nums_int = np.empty((length, ), dtype=object)
    arr_nums_int[:] = list(map(lambda x: tuple(map(int, x)), arr_nums_unique))
    # df_int = pd.DataFrame(data={'val': arr_vals, 'nums': arr_nums_int}, columns=['nums', 'val'])
    # df_int_nums = df_int.set_index('nums')
    df_int_nums = pd.Series(data=arr_vals, index=arr_nums_int)

    d_uint8 = {k: v for k, v in zip(arr_nums_uint8, arr_vals)}
    d_int = {k: v for k, v in zip(arr_nums_int, arr_vals)}


    def mapping_test_df_uint8(l_tpl_nums):
        return df_uint8_nums.loc[l_tpl_nums].values
        # return df_uint8_nums.loc[l_tpl_nums]['val'].values

    def mapping_test_df_int(l_tpl_nums):
        return df_int_nums.loc[l_tpl_nums].values
        # return df_int_nums.loc[l_tpl_nums]['val'].values

    def mapping_test_d_uint8(l_tpl_nums):
        return [d_uint8[t] for t in l_tpl_nums]

    def mapping_test_d_int(l_tpl_nums):
        return [d_int[t] for t in l_tpl_nums]


    n_test = 300000
    arr_idxs = np.random.randint(0, length, (n_test, ))

    arr_idxs_vals = arr_vals[arr_idxs]
    l_idxs_vals = arr_idxs_vals.tolist()
    arr_nums_uint8_idxs = arr_nums_uint8[arr_idxs]
    arr_nums_int_idxs = arr_nums_int[arr_idxs]

    print('Test df_uint8')
    time_df_uint8, ret_l_idxs_vals_df_uint8 = func_timer(mapping_test_df_uint8, [arr_nums_uint8_idxs])
    print('Test df_int')
    time_df_int, ret_l_idxs_vals_df_int = func_timer(mapping_test_df_int, [arr_nums_int_idxs])
    print('Test d_uint8')
    time_d_uint8, ret_l_idxs_vals_d_uint8 = func_timer(mapping_test_d_uint8, [arr_nums_uint8_idxs])
    print('Test d_int')
    time_d_int, ret_l_idxs_vals_d_int = func_timer(mapping_test_d_int, [arr_nums_int_idxs])

    assert np.all(ret_l_idxs_vals_df_uint8 == arr_idxs_vals)
    assert np.all(ret_l_idxs_vals_df_int == arr_idxs_vals)
    assert ret_l_idxs_vals_d_uint8 == l_idxs_vals
    assert ret_l_idxs_vals_d_int == l_idxs_vals

    print("time_df_uint8: {}".format(time_df_uint8))
    print("time_df_int: {}".format(time_df_int))
    print("time_d_uint8: {}".format(time_d_uint8))
    print("time_d_int: {}".format(time_d_int))

    # l_nums_val_df_uint8 = [(arr_nums_uint8[i], arr_vals[i]) for i in arr_idxs]
    # l_nums_val_df_int = [(arr_nums_int[i], arr_vals[i]) for i in arr_idxs]
    
    # l_nums_val_d_uint8 = [(arr_nums_uint8[i], arr_vals[i]) for i in arr_idxs]
    # l_nums_val_d_int = [(arr_nums_int[i], arr_vals[i]) for i in arr_idxs]

    # def convert_num_to_8bit_array(num, dimension):
    #     l = []
    #     while num > 0:
    #         l.append(num % 256)
    #         num //= 256
    #     return l
    # arr_vals = [convert_num_to_8bit_array(num, dimension) for num in l_vals]
