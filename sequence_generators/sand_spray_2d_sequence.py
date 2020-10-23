#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

# TODO: need to be python convension confirm! (coding standard as much as possible!)

import datetime
import dill
import gzip
import os
import pdb
import shutil
import string
import sys

from copy import deepcopy

import matplotlib.pyplot as plt

import numpy as np

from dotmap import DotMap

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization


if __name__ == "__main__":
    print('Hello World!')

    def convert_d_to_arr(d, n):
        arr = np.zeros((n, n), dtype=np.int)
        for (x, y), v in d.items():
            arr[y-1, x-1] = v
        return arr

    d = {(1, 1): 1}
    i = 0
    print("i: {}, d: {}".format(i, d))
    arr = convert_d_to_arr(d, i+1)
    print("arr:\n{}".format(arr))

    for i in range(1, 20):
        d_spray = {}
        for (x, y), v in d.items():
            if v==0:
                continue

            t = (x, y)
            if not t in d_spray:
                d_spray[t] = 0
            d_spray[t] += v-1

            for diff_x, diff_y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                if x+diff_x<1 or y+diff_y<1:
                    continue
                t = (x+diff_x, y+diff_y)
                if not t in d_spray:
                    d_spray[t] = 0
                d_spray[t] += 1

        d = d_spray

        print("i: {}, d: {}".format(i, d))
        # print("d_spray: {}".format(d_spray))
        arr = convert_d_to_arr(d, i+1)
        print("arr:\n{}".format(arr))
