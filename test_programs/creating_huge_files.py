#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import pdb
import sys

# Needed for DataFrame (databases etc.)
import numpy as np
import pandas as pd
import pandas.io.sql as sql

from functools import reduce

from dotmap import DotMap

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

USER_HOME_PATH = os.path.expanduser('~')+'/'
print("USER_HOME_PATH: {}".format(USER_HOME_PATH))

if __name__ == "__main__":
    file_path_obj = USER_HOME_PATH+"Documents/filepath_root.pkl.gz"
    def create_path_obj():
        return {'filepath_root': '/media/doublepmcl/exfat123/'}
        # return {'filepath_root': '/media/doublepmcl/4393-62E6/'}
    obj = get_pkl_gz_obj(create_path_obj, file_path_obj)
    print("obj['filepath_root']: {}".format(obj['filepath_root']))
    # print("Hello World!")

    arr1 = np.arange(0, 0x100).astype(np.uint8)
    arr2 = np.array([np.roll(arr1, i) for i in range(0, 0x100)]).reshape((-1, ))
    # arr = np.repeat(np.arange(0, 0x100).astype(np.uint8), 0x100*0x100/2**4).reshape((0x100, -1)).T.reshape((-1, ))
    arr = np.tile(arr2, 2**26//arr2.shape[0])
    print("arr.shape: {}".format(arr.shape))
    # sys.exit(0)

    times = 2**4*2**4
    with open(obj['filepath_root']+'test_123.hex', 'wb') as f:
        for i in range(0, times):
            print("i: 0x{:03X}".format(i))
            np.roll(arr, i).tofile(f)

    file_size = arr.shape[0]*times
    print("file_size: {}".format(file_size))
    print("file_size/2**20: {}".format(file_size/2**20))

