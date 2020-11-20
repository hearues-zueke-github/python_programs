#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from functools import reduce
from dotmap import DotMap

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__ == '__main__':
    # base1 = 3, convert num to base2 = 2
    base1 = 5
    l_consecutive_ones = []
    l_amount_ones = []
    for n in range(0, 31):
        arr=np.array(list(map(int, bin(base1**n)[2:])))
        print("n: {}, arr: {}".format(n, arr))

        first_i = 0
        last_i = 0
        prev_1 = False
        max_conecutive_length_ones = 0
        for i, v in enumerate(arr, 0):
            if prev_1==False and v==1:
                prev_1 = True
                first_i = i
                last_i = i
            elif prev_1==True and v==1:
                last_i = i
            elif v==0:
                prev_1 = False
            diff_i = last_i-first_i+1
            if max_conecutive_length_ones<diff_i:
                max_conecutive_length_ones = diff_i

        l_consecutive_ones.append(max_conecutive_length_ones)
        l_amount_ones.append(np.sum(arr==1))

        print("- max_conecutive_length_ones: {}".format(max_conecutive_length_ones))
    
    print()
    print("l_consecutive_ones: {}".format(str(l_consecutive_ones).replace(', ', ' ')))
    print("l_amount_ones: {}".format(str(l_amount_ones).replace(', ', ' ')))
