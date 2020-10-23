#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import os
import sys
import time

import numpy as np

from copy import deepcopy
from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

import utils_sequence

sys.path.append("../math_numbers")
import prime_numbers_fun

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

np.set_printoptions(threshold=sys.maxsize)


if __name__=='__main__':
    n = 10
    print("n: {}".format(n))
    l = list(range(0, n))
    # print("l: {}".format(l))

    l_list = []
    for jump in range(1, n+1):
        # print("jump: {}".format(jump))
        # for i_start in range(0, n):

        l_c = deepcopy(l)
        l_j = []

        i = 0
        # i = i_start
        for j in range(n, 0, -1):
            i = (i+jump-1)%j
            l_j.append(l_c.pop(i))
        # print("  l_j: {}".format(l_j))

        l_list.append(l_j)

        # print("  i_start: {}, l_j: {}".format(i_start, l_j))
    arr_list = np.array(l_list)
    arr_diff_mod = (arr_list-np.roll(arr_list, 1, axis=1))%n
    print("arr_diff_mod:\n{}".format(arr_diff_mod))
