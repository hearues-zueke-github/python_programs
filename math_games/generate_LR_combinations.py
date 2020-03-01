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

    globals()['l_list_tpl'] = l_list_tpl
    
    if remove_duplicate_moves:
        # pass
        l_factors = get_possible_factors(max_depth)
        for factor in l_factors:
            factor_multiply = max_depth//factor
            s_remove_duplicates = {tpl*factor_multiply for tpl in l_list_tpl[factor-1]}
            s_move_string_tpl -= s_remove_duplicates

        # l_remove_idx = []
        # for i, move_tpl in enumerate(s_move_string_tpl, 0):
        #     for j in l_factors:
        #         l_tpls = [move_tpl[j*k:j*(k+1)] for k in range(0, max_depth//j)]
        #         tpl_first = l_tpls[0]
        #         if all([tpl==tpl_first for tpl in l_tpls[1:]]):
        #             l_remove_idx.append(i)
        #             break
        # print("l_remove_idx: {}".format(l_remove_idx))
        # for i in reversed(l_remove_idx):
        #     s_move_string_tpl.pop(i)

    l_move_string = [''.join(tpl) for tpl in s_move_string_tpl]

    return l_move_string


if __name__=='__main__':
    # l_base_strings = []
    # for i in range(2, max_length+1):
    #     for j in range(1, i):
    #         l_base_strings.append('L'*j+'R'*(i-j))

    max_length = 4
    l_base_string = get_base_strings(max_length=max_length)

    max_depth = 6
    # l_move_string = get_combined_base_strings(l_base_string, max_depth=max_depth, remove_duplicate_moves=False)
    l_move_string = get_combined_base_strings(l_base_string, max_depth=max_depth, remove_duplicate_moves=True)

    print("max_length: {}".format(max_length))
    print("max_depth: {}".format(max_depth))
    print("len(l_base_string): {}".format(len(l_base_string)))
    print("len(l_move_string): {}".format(len(l_move_string)))

    sys.exit(0)

    l_move_strings = []
    l_move_strings.extend(l_base_strings)

    s_base_string = set(l_base_strings)
    for move1 in s_base_string:
        for move2 in s_base_string:
            l_move_strings.append(move1+move2)
            # for move3 in s_base_string:
            #     l_move_strings.append(move1+move2+move3)

    l_len = len(l_move_strings)
    l_move_strings = l_move_strings[:l_len-l_len%4]
    print("{}".format(';'.join(l_move_strings)))
