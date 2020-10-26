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
    # n = 7
    l_a = []
    d_n_moves = {}
    for n in range(4, 41):
        print("n: {}".format(n))
        l_pos = [(j, i) for j in range(0, n) for i in range(0, n)]
        s_pos = set(l_pos)

        l_pos_now = [(0, 0)]
        s_pos.remove(l_pos_now[0])

        d_moves = {0: l_pos_now}

        l_y_x_deltas = [
            (2, 1), (-2, 1), (2, -1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2),
        ]
        moves = 1
        while len(s_pos) > 0:
            l_pos_new = []
            for (y, x) in l_pos_now:
                for dy, dx in l_y_x_deltas:
                    t = (y+dy, x+dx)
                    if t in s_pos:
                        s_pos.remove(t)
                        l_pos_new.append(t)

            l_pos_now = sorted(l_pos_new)
            d_moves[moves] = l_pos_now
            moves += 1

        print("- d_moves: {}".format(d_moves))
        print("- moves: {}".format(moves))
        l_a.append(moves)
        d_n_moves[n] = d_moves

    print("l_a: {}".format(l_a))
    ll = [[len(l) for l in d_n_moves[n].values()] for n in sorted(d_n_moves.keys())]
    l_max = [max(l) for l in ll]
    print("l_max: {}".format(l_max))
