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

sys.path.append('../combinatorics')
from different_combinations import get_all_combinations_repeat

import numpy as np
import multiprocessing as mp

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

if __name__ == '__main__':
    l_digits = [0, 1]
    l = [0, 1]

    l_t_all_rest = []
    l_t_all_rest += list(map(tuple, get_all_combinations_repeat(2, 1).tolist()))
    l_t_all_rest += list(map(tuple, get_all_combinations_repeat(2, 2).tolist()))

    for iteration in range(0, 30):
        print("iteration: {}".format(iteration))
        # combine all current combinations of tuples!
        l_t = []
        
        for i1 in range(0, len(l)):
            for i2 in range(i1+1, len(l)+1):
                l_t.append(tuple(l[i1:i2]))

        l_t_sort = sorted(set(l_t), key=lambda x: (len(x), x))

        for t in l_t_sort:
            if t in l_t_all_rest:
                l_t_all_rest.remove(t)

        n_current = 2

        l_t_next = []
        for i in range(len(l)-1, -1, -1):
            t = tuple(l[i:])
            if iteration%2==0:
                for v in l_digits[::-1]:
                    l_t_next.append(t+(v, ))
            else:
                for v in l_digits:
                    l_t_next.append(t+(v, ))

        for t in l_t_next:
            if len(t) > n_current:
                n_current += 1
                l_t_all_rest += list(map(tuple, get_all_combinations_repeat(2, n_current).tolist()))
            if t in l_t_all_rest:
                print("Found next t: {}".format(t))
                l_t_all_rest.remove(t)
                l.append(t[-1])
                break

        print("- l_t: {}".format(l_t))
        print("- l_t_sort: {}".format(l_t_sort))
        print("- l_t_all_rest: {}".format(l_t_all_rest))
        print("- l_t_next: {}".format(l_t_next))
        print("- l: {}".format(l))
