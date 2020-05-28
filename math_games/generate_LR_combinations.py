#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import random
import sys

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

def get_possible_factors(n):
    l = []
    for i in range(1, n):
        if n%i==0:
            l.append(i)
    return l


def get_base_strings(max_length):
    l_base_strings = []
    max_length = 4
    for i in range(2, max_length+1):
        for j in range(1, i):
            l_base_strings.append('L'*j+'R'*(i-j))
    return l_base_strings


def get_combined_base_strings(l_base_string, max_depth=2, remove_duplicate_moves=True):
    s_base_string_tpl = {(move, ) for move in l_base_string}
    s_move_string_tpl = deepcopy(s_base_string_tpl)
    s_move_string_tpl_new = set()
    l_list_tpl = [s_base_string_tpl]
    for depth in range(1, max_depth):
        for move1 in s_move_string_tpl:
            for move2 in s_base_string_tpl:
                s_move_string_tpl_new.add(move1+move2)
        s_move_string_tpl = s_move_string_tpl_new
        l_list_tpl.append(s_move_string_tpl)
        s_move_string_tpl_new = set()

    if remove_duplicate_moves:
        l_factors = get_possible_factors(max_depth)
        for factor in l_factors:
            factor_multiply = max_depth//factor
            s_remove_duplicates = {tpl*factor_multiply for tpl in l_list_tpl[factor-1]}
            s_move_string_tpl -= s_remove_duplicates

    l_move_string = [''.join(tpl) for tpl in s_move_string_tpl]

    return l_move_string


def get_all_moves_string(max_base_length, max_depth=2, remove_duplicate_moves=True):
    l_base_string = get_base_strings(max_length=max_base_length)
    l_move_string = get_combined_base_strings(l_base_string, max_depth=max_depth, remove_duplicate_moves=True)

    return l_move_string


if __name__=='__main__':
    max_base_length = 4
    max_depth = 6
    l_move_string = get_all_moves_string(max_base_length=max_base_length, max_depth=max_depth, remove_duplicate_moves=True)

    print("{}".format(';'.join(l_move_string)))
