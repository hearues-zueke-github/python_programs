#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import sys

import matplotlib.pyplot as plt

import numpy as np

from copy import deepcopy
from math import factorial as fac

sys.path.append("../combinatorics/")
from different_combinations import get_all_combinations_repeat, get_permutation_table

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

def write_digraph_as_dotfile(path, arr_x, arr_y):
    with open(path, 'w') as f:
        f.write('digraph {\n')
        for x in arr_x:
            f.write(f'  x{x}[label="{x}"];\n')
        f.write('\n')
        for x, y in zip(arr_x, arr_y):
            f.write(f'  x{x} -> x{y};\n')

        f.write('}\n')


if __name__=='__main__':
    n = 100
    l = list(range(1, n+1))

    i = 0
    i2 = 1
    l2 = []
    while i<len(l):
        l2.append(l.pop(i))
        if i>i2:
            i -= i2+1
        else:
            i += i2
        i2 += 1
    # l2 = sorted(l2)
    print("l2: {}".format(l2))

    arr2 = np.array(l2)
    arr2_sort = np.sort(arr2)

    idx = np.where(np.diff(arr2_sort)>1)[0][0]
    print("idx: {}".format(idx))
    print("arr2_sort[idx]: {}".format(arr2_sort[idx]))
    print("arr2_sort[idx+1]: {}".format(arr2_sort[idx+1]))

    # write_digraph_as_dotfile(path='collatz_1_graph_n_{}.dot'.format(n), arr_x=arr_x, arr_y=arr_y)
