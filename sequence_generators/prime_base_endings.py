#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import datetime
import dill
import gzip
import inspect
import os
import pdb
import shutil
import string
import sys

from copy import deepcopy

import matplotlib.pyplot as plt

import numpy as np
from array2gif import write_gif

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

from utils_sequence import check_if_prime, num_to_base

def base_rev_to_num(l, b):
    e = b**(len(l)-1)
    s = 0
    for i in l:
        s += e*i
        e //= b
    return s


PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization

sys.path.append(PATH_ROOT_DIR+'/../math_numbers')
from prime_numbers_fun import get_primes


if __name__ == "__main__":
    l_primes = [i for i in get_primes(100000)]        
    # print("l_primes: {}".format(l_primes))

    b = 13
    l_primes_base = [num_to_base(i, b)[::-1] for i in l_primes]

    # base_length = 4
    for base_length in range(1, 3):
        l_primes_endings = [[0]*(0 if len(l)>=base_length else base_length-len(l))+l[-base_length:] for l in l_primes_base]
        l_primes_endings_num = [base_rev_to_num(l, b) for l in l_primes_endings]
        u, c = np.unique(l_primes_endings_num, return_counts=True)
        print("b: {}, base_length: {}".format(b, base_length))
        print("u: {}".format(u))
        print("c: {}".format(c))
