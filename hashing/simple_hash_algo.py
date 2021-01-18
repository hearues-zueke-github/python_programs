#! /usr/bin/python3

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

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':
    # print("Hello World!")

    # n = 1234534
    # modulo = 314

    # n_div = n // modulo
    # n_mod = n % modulo

    # print("n: {}".format(n))
    # print("modulo: {}".format(modulo))
    # print("n_div: {}".format(n_div))
    # print("n_mod: {}".format(n_mod))

    # l_div = []
    # l_mod = []

    # modulo = 7
    # print("modulo: {}".format(modulo))

    # for i in range(0, 20):
    #     v_div = i // modulo
    #     v_mod = i % modulo
    #     l_div.append(v_div)
    #     l_mod.append(v_mod)

    #     # print("i: {:3}, v_div: {:2}, v_mod: {:2}".format(i, v_div, v_mod))
    #     print("{:3}: ({:2}, {:2})".format(i, v_div, v_mod))

    seed = 4

    a = 5
    b = 7

    x = seed
    modulo = 15
    l = [x]

    for _ in range(0, 30):
        x = (a * x + b) % modulo
        l.append(x)

    print("seed: {}, a: {}, b: {}".format(seed, a, b, ))
    print("l: {}".format(l))
