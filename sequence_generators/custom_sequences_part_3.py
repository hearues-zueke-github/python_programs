#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import io
import datetime
import os
import pathlib
import re
import shutil
import string
import subprocess
import sys
import time
import mmap

import numpy as np
import multiprocessing as mp

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

def get_func_f(p1=1, p2=1, base=10):
    def a(n):
        if n < 0:
            return 0

        if n in a.d:
            return a.d[n]
        if a.n_last < n:
            for i in range(a.n_last+1, n+1):
                v = f(n, 0, 0)
                a.d[n] = v
            a.n_last = n
        return a.d[n]


    def f(n, i, j):
        t = (n, i, j)
        if t in f.d:
            return f.d[t]

        if n <= 0:
            v = (j + i) % base
            # v = j % base
            # v = j
            # v = (i+j) % base
            f.d[t] = v
            return v

        # v1 = a(n-1)
        # v = ((((n**2+1)%base)+v1)**2+1) % base
        # v = f(n - v - j - 1 - i, i + j + p2 + 1, (j + v)%base)
        
        v1 = a(n-1-p1)
        v2 = a(n-1-p2)
        v = (f(n - i - v1 - 1, i + v1 + 1, (j + 1) % base) + v1 + v2) % base
        # v = n % base
        
        # v2 = a(n-2)+1*p1
        # v3 = a(n-3)+2*p2
        # v = f(n - v1 - 1 - i*p1 - j - p2, i + v1 + p1 + j + 1, j + 1)
        f.d[t] = v
        return v

    # def f(n, i, j, k):
    #     t = (n, i, j)
    #     if t in f.d:
    #         return f.d[t]

    #     if n <= 0:
    #         v = (i+j) % base
    #         f.d[t] = v
    #         return v

    #     v1 = a(n-1)
    #     v2 = a(n-2)+1*p1
    #     v3 = a(n-3)+2*p2
    #     v = f(n - v1 - 1 - i*p1 - j, i + v1 + 1, j + v2 + k + 1, k + v3 + 1)
    #     f.d[t] = v
    #     return v

    a.d = {0: (p1+p2)%base}
    a.n_last = 0
    f.d = {}

    return a


if __name__ == '__main__':
    print('Hello World!')
    n_max = 10000
    base = 20
    max_p2 = 30
    l_p2 = list(range(0, max_p2))
    d_p2 = {}
    for p2 in l_p2:
        print("p2: {}".format(p2))
        a = get_func_f(p1=0, p2=p2, base=base)
        # a = get_func_f(p2=p2, base=base)
        l = [a(i) for i in range(0, n_max+1)]
        d_p2[p2] = l

    # # plots
    # nrows = 5
    # fig, axs = plt.subplots(nrows=nrows, figsize=(12, 10))

    # plt.suptitle('n_max: {}, base: {}'.format(n_max, base))
    
    # xs = np.arange(0, len(d_p2[1]))

    # ms = 5
    # marker = '.'
    # # marker = (4, 0, 45)

    # for i in range(0, nrows):
    #     axs[i].plot(xs, d_p2[i+1], ls='', marker=marker, ms=ms)
    #     axs[i].set_title('p2 = {}'.format(i+1))

    # plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.1)

    # plt.show()


    dir_pics = TEMP_DIR+'pictures/sequences/'
    if not os.path.exists(dir_pics):
        os.makedirs(dir_pics)
    print("dir_pics: {}".format(dir_pics))

    # create the image of the modulus!
    l_pix_part = []
    
    for p2 in l_p2:
        pix_empty = np.zeros((n_max+1, 1), dtype=np.uint8)+2
        pix_part = np.zeros((n_max+1, base), dtype=np.uint8)
        pix_part[np.arange(0, n_max+1), d_p2[p2]] = 1
        l_pix_part.extend([pix_empty, pix_part])

    l_pix_part.pop(0)

    arr = np.hstack(l_pix_part)
    colors = np.array([
        [0x00, 0x00, 0x00],
        [0xFF, 0xFF, 0xFF],
        [0x80, 0x80, 0x80],
    ], dtype=np.uint8)

    pix = colors[arr]
    img = Image.fromarray(pix)
    # img.show()

    img.save(dir_pics+'custom_sequence_n_max_{}_base_{}_nr_3.png'.format(n_max, base))
