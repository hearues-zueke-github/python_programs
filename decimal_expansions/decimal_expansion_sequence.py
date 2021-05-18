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
import multiprocessing as mp

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union
from PIL import Image

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

import importlib.util as imp_util

# TODO: change the optaining of the git root folder from a config file first!
spec = imp_util.spec_from_file_location("utils", os.path.join(HOME_DIR, "git/python_programs/utils.py"))
utils = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils)

mkdirs = utils.mkdirs

spec = imp_util.spec_from_file_location("utils_multiprocessing_manager", os.path.join(HOME_DIR, "git/python_programs/utils_multiprocessing_manager.py"))
utils_multiprocessing_manager = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils_multiprocessing_manager)

MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

from decimal import Decimal as Dec, getcontext
import math

import mpmath

def convert_n_to_base(n: int, base: int) -> List[int]:
    if n == 0:
        return [0]

    l_n: List[int] = []

    while n > 0:
        l_n.append(n % base)
        n //= base

    return l_n[::-1]


class Number(Exception):
    __slots__ = ['n', 'base', 'l_n']

    def __init__(self, n, base):
        self.n = n
        self.base = base
        self.l_n = convert_n_to_base(n, base)

    def __repr__(self):
        return f"Test(n={self.n}, base={self.base}, l_n={self.l_n})"

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    print("Hello World!")
    getcontext().prec = 100

    num = Number(3, 5)

    sys.exit()

    l_length_seq = []
    d_denominator = {}
    for n_den in range(2, 201):
        d_nominator = {}
        for n_nom in range(1, 2):
        # for n_nom in range(1, n_den):
            getcontext().prec = n_den * 10

            dec_nom = Dec(n_nom)
            dec_den = Dec(n_den)
            dec_number = dec_nom / dec_den
            d_nominator[n_nom] = dec_number

            s = str(dec_number)
            l = list(s[2:-1])[::-1]

            # find repeating pattern!
            is_found = False
            length = len(l)
            for i in range(1, length//2):
                if l[:i] == l[i:i*2]:
                    is_found = True
                    break

            if is_found:
                print('nom: {}, den: {}, i: {}, l[:i]: {}, s: {}'.format(n_nom, n_den, i, l[:i], s))
                if i == n_den - 1:
                    l_length_seq.append((n_den, i))
            else:
                print('s: {}'.format(s))
                # l_length_seq.append((n_den, None))

        d_denominator[n_den] = d_nominator
