#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import traceback

import numpy as np
import pandas as pd

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile

sys.path.append('..')
from utils import mkdirs

from random import randint

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':
    print("Hello World!")

    # 0 : green
    # 1 - 18 : red
    # 17 - 36 : black

    l = []

    for _ in range(0, 1000):
        v = randint(0, 37)
        for _ in range(0, 10):
            v = (v + randint(0, 37)) % 37

        l.append(v)

    print("l: {}".format(l))

    d_color = {**{0: 'g'}, **{i: 'r' for i in range(1, 19)}, **{i: 'k' for i in range(19, 37)}}
    l_color = [d_color[v] for v in l]

    print("l_color: {}".format(l_color))

    amount = 0
    in_game = 1
    prev_equal = True
    v = l_color[0]

    l_amount = []
    l_in_game = []
    l_prev_equal = []

    for v_next in l_color[1:]:
        if v != v_next:
            amount -= in_game
            in_game *= 2
            prev_equal = False
        else:
            if prev_equal:
                amount += 2
                in_game = 1
            else:
                amount += 2*in_game
                in_game = 1
                prev_equal = True

        l_amount.append(amount)
        l_in_game.append(in_game)
        l_prev_equal.append(prev_equal)

    print("l_amount: {}".format(l_amount))
