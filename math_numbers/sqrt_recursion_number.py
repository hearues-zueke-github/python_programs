#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

# sys.path.append("../combinatorics/")
# import different_combinations as combinations
# sys.path.append("../math_numbers/")
# from prime_numbers_fun import get_primes

from time import time

from PIL import Image

import numpy as np

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

import decimal
decimal.getcontext().prec = 100

from decimal import Decimal as Dec

if __name__ == "__main__":
    print("Hello World!")
    n = 10000
    s = np.sqrt(Dec(n))
    for i in range(n-1, 0, -1):
        s = np.sqrt(Dec(i)-s)
        print("i: {}, s: {}".format(i, s))
