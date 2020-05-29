#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

# TODO: need to be python convension confirm! (coding standard as much as possible!)

import datetime
import dill
import gzip
import inspect
import os
import pdb
import shutil
import string
import sys

from copy import deepcopy

import matplotlib.pyplot as plt

import numpy as np
from array2gif import write_gif

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization


def num_to_base(n, b):
    l = []
    while n>0:
        l.append(n%b)
        n = n//b
    return l


def digsum(n, b):
    s = 0
    while n>0:
        s += n%b
        n = n//b
    return s


def get_list_of_limits(n):
    s = n
    l_inc = [s]
    for j in range(n-1, 1, -1):
        if s%j>0:
            s += j-(s%j)
        l_inc.append(s)
    return l_inc[::-1]


if __name__ == "__main__":
    # l = []
    # l_increments = []
    # for i in range(2, 20):
    #     l_inc = get_list_of_limits(i)
    #     l.append(l_inc[0])
    #     l_increments.append(l_inc)
    # print("l: {}".format(l))
    # print("l_increments: {}".format(l_increments))

    # sys.exit(0)

    n_max = 200
    l_inc = get_list_of_limits(n_max)
    print("n_max: {}".format(n_max))
    # print("l_inc: {}".format(l_inc))
    n = l_inc[0]
    l = [i for i in range(0, n)]
    l_1 = [l[0]]

    # for j, n_ in enumerate(l_inc, 1):
    #     print("j: {}, n_: {}".format(j, n_))
    #     y = 0
    #     while y+j<n_:
    #         for i in range(0, j):
    #             i1 = y
    #             i2 = y+j
    #             l[i1], l[i2] = l[i2], l[i1]
    #             y += 1
    #         y += j
    #     l_1.append(l[0])

    for j, n_ in enumerate(l_inc, 2):
        # print("j: {}, n_: {}".format(j, n_))
        y = 0
        while y<n_:
            t = l[y]
            for i in range(y, y+j-1):
                l[i] = l[i+1]
            l[y+j-1] = t
            y += j
            # print("y: {}".format(y))
        # print("l: {}".format(l))
        l_1.append(l[0])

    plt.figure()

    plt.plot(np.arange(0, len(l_1)), l_1, '.b')

    plt.show(block=False)

    print("l_1: {}".format(l_1))

