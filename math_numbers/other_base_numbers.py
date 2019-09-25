#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

# sys.path.append("../combinatorics/")
# from different_combinations import get_permutation_table
# import different_combinations
# from prime_numbers_fun import get_primes

import numpy as np

from PIL import Image
from functools import reduce
from math import factorial
from time import time

sys.path.append("..")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

def convert_int_to_base(n, b):
    lst = []
    while n > 0:
        lst.append(n%b)
        n //= b
    return lst[::-1]

if __name__ == "__main__":
    # n = 102
    # n_b_2 = convert_int_to_base(n, 2)
    # n_b_2_str = bin(n)[2:]

    # print("n: {}".format(n))
    # print("n_b_2: {}".format(n_b_2))
    # print("n_b_2_str: {}".format(n_b_2_str))
    nums = []
    for n in range(2, 101):
        # print("n: {}".format(n))
        lsts = [(i, convert_int_to_base(n, i)) for i in range(2, n+1)]
        lst_digit_sum = [(i, np.sum(l)) for i, l in lsts]
        # print("lsts: {}".format(lsts))
        # print("lst_digit_sum: {}".format(lst_digit_sum))
        a = np.array(lst_digit_sum)
        # arr_digit_sum = np.array(lst_digit_sum)
        # print("arr_digit_sum:\n{}".format(arr_digit_sum))
        b = a[:,0][np.where(a[:,0]>a[:,1])[0]]
        if b.shape[0]==0:
            nums.append(0)
        else:
            nums.append(b[0])
    
    # sequence A321909, missing [2, 2] at the beginning
    print("nums: {}".format(nums))
