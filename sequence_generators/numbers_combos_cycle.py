#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import sys
import time

import numpy as np

from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

import utils_sequence

if __name__ == "__main__":
    print("Hello World!")

    m = 256
    n = 3

    length = 2*(m-1)*m**(n-1)
    print("length: {}".format(length))

    arr = np.zeros((length, n), dtype=np.int)

    for i in range(1, n+1):
        if i == n:
            arr_resh = arr.reshape((2*(m-1), m**(i-1), n))
            arr_resh[:m, :, -i] = np.arange(0, m).reshape((-1, 1))
            arr_resh[m:m+m-2, :, -i] = np.arange(m-2, 0, -1).reshape((-1, 1))
        else:
            arr_resh = arr.reshape((-1, m, m**(i-1), n))
            idxs1 = np.arange(0, arr_resh.shape[0], 2)
            idxs2 = np.arange(1, arr_resh.shape[0], 2)
            arr_resh[idxs1, :, :, -i] = np.arange(0, m).reshape((-1, 1))
            arr_resh[idxs2, :, :, -i] = np.arange(m-1, -1, -1).reshape((-1, 1))

    # TODO: create a image with the pixels!

    # length = arr.shape[0]
    l1 = int(np.sqrt(length))
    if l1**2 < length:
        l2 = l1+1
    print("l1: {}".format(l1))
    print("l2: {}".format(l2))
    print("l1*l2: {}".format(l1*l2))

    arr2 = np.vstack((arr.astype(np.uint8), np.zeros((l1*l2-length, 3), dtype=np.uint8)))
    arr3 = arr2.reshape((l1, l2, 3))
    
    idxs = np.arange(1, l1, 2)
    arr3[idxs] = np.flip(arr3[idxs], 1)

    img = Image.fromarray(arr3)
    img.save("numbers_combos.png")

    print("arr:\n{}".format(arr))
