#! /usr/bin/python3

# -*- coding: utf-8 -*-

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
    print('Hello World!')
    l_primes = list(get_primes(100))[:6]
    l_sums = []
    l_t = []
    for i1, p1 in enumerate(l_primes, 0):
        for i2, p2 in enumerate(l_primes[i1+1:], i1+1):
            for i3, p3 in enumerate(l_primes[i2+1:], i2+1):
                l_sums.append(p1*p2+p3)
                l_t.append(((i1, i2, i3), (p1, p2, p3)))

    arr = np.array(sorted(set(l_sums)))
    print("arr: {}".format(arr))
