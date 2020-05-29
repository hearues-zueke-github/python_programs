#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import sys

import matplotlib.pyplot as plt

import numpy as np

from math import factorial as fac

sys.path.append("../combinatorics/")
from different_combinations import get_permutation_table

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__=='__main__':
    n = 5
    arr_pt = get_permutation_table(n)
    print("arr_pt.shape:\n{}".format(arr_pt.shape))
    # print('Hello World!')

    d_stat = {}
    for row1 in arr_pt:
        a1 = row1.copy()
        
        for row2 in arr_pt:
            a2 = row2.copy()

            t = tuple(a1.tolist())+tuple(a2.tolist())
            l = [t]
            while True:
                a3 = a1[a2].copy()
                a1 = a2
                a2 = a3
                t = tuple(a1.tolist())+tuple(a2.tolist())
                # t = tuple(a.tolist())
                if t in l:
                    l.append(t)
                    break
                l.append(t)

            length = len(l)
            if not length in d_stat:
                d_stat[length] = []
            d_stat[length].append(l)

    print("d_stat.keys(): {}".format(d_stat.keys()))

    d_stat_true_cycle_length = {}
    for v in d_stat.values():
        for l in v:
            length = len(l)-l.index(l[-1])-1
            if not length in d_stat_true_cycle_length:
                d_stat_true_cycle_length[length] = []
            d_stat_true_cycle_length[length].append(l[-1])

    l_cycle_lengths = sorted([(k, len(v)) for k, v in d_stat_true_cycle_length.items()])
    print("l_cycle_lengths: {}".format(l_cycle_lengths))
    print("len(l_cycle_lengths): {}".format(len(l_cycle_lengths)))
