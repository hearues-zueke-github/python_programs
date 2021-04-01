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

from typing import List

sys.path.append('..')
from utils import mkdirs

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

def convert_num_to_l_base(n: int, b: int) -> List[int]:
    l: List[int] = []

    while n > 0:
        l.append(n % b)
        n //= b

    return l


def convert_l_base_to_num(l: List[int], b: int) -> int:
    p: int = 1
    s: int = 0
    
    v: int
    for v in l:
        s += v * p
        p *= b

    return s


if __name__ == '__main__':
    # print("Hello World!")

    s_hex = '02FE490A'
    l_b_str = [s_hex[i:i+2] for i in range(0, len(s_hex), 2)]
    b = list(map(lambda x: int(x, 16), l_b_str))
    # b = b'\x12\x34\x54\xF8'
    l = list(map(int, b))

    assert all([(x >= 0) and (x <= 255) for x in l])

    # print("b: {}".format(b))
    print("l: [{}]".format(', '.join(map(lambda x: '{:02X}'.format(x), l))))
    
    l_1 = l

    # l_values = []

    f_n2l = convert_num_to_l_base
    f_l2n = convert_l_base_to_num

    n_1 = f_l2n(l_1, 256)
    # print("n_1: {}".format(n_1))

    l_2 = f_n2l(n_1, 255)

    print("l_1: {}".format(l_1))
    print("l_2: {}".format(l_2))
