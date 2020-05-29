#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

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

def get_basic_permutation_list(n):
    l = list(range(0, n))
    i = 1
    m = n
    lp = [0]*n
    for j in range(0, n):
        i = (i+j)%m
        lp[j] = l.pop(i)
        # lp.append(l.pop(i))
        m -= 1
    return lp


if __name__ == "__main__":
    # l_last_n = []
    # l_same_pos_i = []
    # for i in range(1, 8001):
    #     if i%1000==0:
    #         print("i: {}".format(i))
        
    #     lp = get_basic_permutation_list(i)
    #     if lp[-1]==i-1:
    #         l_last_n.append(i)
        
    #     arr = np.array(lp)
    #     l_same_pos_i.append(np.sum(arr==np.arange(0, i)))
    # print("l_last_n: {}".format(l_last_n))
    # # print("l_same_pos_i: {}".format(l_same_pos_i))

    # l_amount_pos_last_n = [l_same_pos_i[j-1] for j in l_last_n]
    # print("l_amount_pos_last_n: {}".format(l_amount_pos_last_n))

    # sys.exit(0)

    n_max = 5000
    l = list(range(0, n_max))
    l_first = [0]

    l_first_orig = [0]
    l_last_orig = [0]

    # print("{}".format(l))

    for i in range(2, n_max):

        # # mirror from middle
        # for j in range(0, i//2):
        #     l[j], l[i-1-j] = l[i-1-j], l[j]
        
        # # shift to left
        # t = l[0]
        # for j in range(0, i-1):
        #     l[j] = l[j+1]
        # l[i-1] = t

        # # shift to right
        # t = l[i-1]
        # for j in range(i-1, 0, -1):
        #     l[j] = l[j-1]
        # l[0] = t

        lp = get_basic_permutation_list(i)

        l_first_orig.append(lp.index(0))
        l_last_orig.append(lp.index(i-1))

        l_change = [0]*i
        for v, j in zip(l[:i], lp):
            l_change[j] = v
        l[:i] = l_change

        # print("{}".format(l))

        l_first.append(l_change[0])

    # print("l_first: {}".format(l_first))

    plt.figure()
    plt.title('l_first')
    plt.plot(np.arange(0, len(l_first)), l_first, '.b', markersize=3)
    plt.show(block=False)

    plt.figure()
    plt.title('l_first_orig')
    plt.plot(np.arange(0, len(l_first_orig)), l_first_orig, '.b', markersize=3)
    plt.show(block=False)

    plt.figure()
    plt.title('l_last_orig')
    plt.plot(np.arange(0, len(l_last_orig)), l_last_orig, '.b', markersize=3)
    plt.show(block=False)
