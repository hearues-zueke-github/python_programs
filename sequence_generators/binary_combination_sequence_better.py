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

sys.path.append('../combinatorics')
from different_combinations import get_all_combinations_repeat

if __name__ == '__main__':
    print('Hello World!')

    l = [0, 1]
    for n in range(2, 3):
        arr = get_all_combinations_repeat(2, n)
        l_comb = list(map(tuple, arr.tolist()))
        print("l_comb: {}".format(l_comb))

        l_comb_current = [tuple(l[i:i+n]) for i in range(0, len(l)-n+1)]
        t_first = tuple(l[-n:])
        s_comb = (set(l_comb)-set(l_comb_current))|set([t_first])

        # abcde -> next: bcde, prev: abcd
        # d_t_prev =  {}
        d_t_next =  {}
        d_prev_t =  {}
        # d_next_t =  {}
        for t in s_comb:
            t_prev = t[:-1]
            t_next = t[1:]

            # d_t_prev[t] = t_prev
            d_t_next[t] = t_next

            if t_prev not in d_prev_t:
                d_prev_t[t_prev] = []
            d_prev_t[t_prev].append(t)

            # if t_next not in d_next_t:
            #     d_next_t[t_next] = []
            # d_next_t[t_next].append(t)

        d_t_to_l = {t: d_prev_t[next_part] for t, next_part in d_t_next.items()}

        
