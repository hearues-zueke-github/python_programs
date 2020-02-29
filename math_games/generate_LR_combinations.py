#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import random
import sys

from pysat.solvers import Glucose3

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time
from functools import reduce

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"


if __name__=='__main__':
    l_base_strings = ['LR']
    max_length = 8
    for i in range(3, max_length+1):
        for j in range(1, i):
            l_base_strings.append('L'*j+'R'*(i-j))
    # print("{}".format(';'.join(l_base_strings)))
    # print("l_base_strings: {}".format(l_base_strings))

    l_move_strings = []
    l_move_strings.extend(l_base_strings)

    s_base_string = set(l_base_strings)
    s_base_string2 = set(l_base_strings)
    for move1 in s_base_string:
        s_base_string2.remove(move1)
        for move2 in s_base_string2:
            l_move_strings.append(move1+move2)
        s_base_string2.add(move1)

    l_len = len(l_move_strings)
    l_move_strings = l_move_strings[:l_len-l_len%4]
    print("{}".format(';'.join(l_move_strings)))
