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

from decimal import Decimal as Dec, getcontext
import math

import mpmath

sys.path.append('..')
from utils import mkdirs

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':
    print("Hello World!")

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
