#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import os
import sys
import time

import numpy as np

from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

sys.path.append("../math_numbers")
import prime_numbers_fun

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

np.set_printoptions(threshold=sys.maxsize)

if __name__=='__main__':
    a = 5
    c = 7
    n = 16
    x = (a*np.arange(0, n)+c)%n
    print("x: {}".format(x))

    l = []
    i = 0
    for _ in range(0, n):
        i = (a*i+c)%n
        l.append(i)
    print("l: {}".format(l))

    sys.exit(0)

    n = 256

    x = np.arange(0, n)
    v1 = np.tile(x, n).reshape((n, n))

    v2 = np.dstack((v1, v1.T)).reshape((n**2, 2))
    arr_a, arr_c = v2.T

    x_i = np.zeros((n**2, ), dtype=np.int)

    lst_A = []
    for i in range(0, n):
        x_i = (arr_a*x_i+arr_c)%n
        lst_A.append(x_i)

    A = np.array(lst_A).T

    idxs = [np.unique(r).shape[0]==n for r in A]

    arr_a_c = v2[idxs]
    lst_a_c = sorted(list(map(tuple, arr_a_c.tolist())))
    # print("lst_a_c: {}".format(lst_a_c))

    unique_a = np.unique(arr_a_c[:, 0])
    unique_c = np.unique(arr_a_c[:, 1])
    print("unique_a: {}".format(unique_a.tolist()))
    print("unique_c: {}".format(unique_c.tolist()))
