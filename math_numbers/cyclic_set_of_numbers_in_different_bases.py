#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

import matplotlib.pyplot as plt

import numpy as np

from math import factorial as fac

from utils_math_numbers import convert_n_to_other_base, convert_base_n_to_num

sys.path.append("../combinatorics/")
from different_combinations import get_permutation_table

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"


if __name__=='__main__':
    found_cyclic_numbers_in_base = {}

    # for n in range(3, 4):
    for n in range(5, 6):
        print("n: {}".format(n))
        arr = get_permutation_table(n)
        for b in range(n, 1000):
            for row in arr:
                num = convert_base_n_to_num(row, b)
                if num > 10000 or num < 2:
                    continue
                if not num in found_cyclic_numbers_in_base:
                    found_cyclic_numbers_in_base[num] = []
                found_cyclic_numbers_in_base[num].append((b, n, row.tolist()[::-1]))

    print("len(found_cyclic_numbers_in_base): {}".format(len(found_cyclic_numbers_in_base)))

    arr = np.array(sorted(found_cyclic_numbers_in_base.keys()))
    idxs = np.where((arr[1:]-arr[:-1])>1)[0]
    print("idxs: {}".format(idxs))

    if idxs.shape[0]>0:
        print("idxs[0]: {}".format(idxs[0]))
        print("arr[idxs[0]]: {}".format(arr[idxs[0]]))

    l_lens = [len(found_cyclic_numbers_in_base[k]) for k in sorted(found_cyclic_numbers_in_base.keys())]
    print("l_lens: {}".format(l_lens[:100]))
