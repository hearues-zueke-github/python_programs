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
    n = 100000
    l = list(range(1, n+1))

    i = 2
    # i2 = 1
    l2 = [1]
    while True:
        print("i: {}".format(i))
        print("l: {}".format(l))
        part_idxs = l[:i]
        print("- part_idxs: {}".format(part_idxs))
        idx = np.sum(part_idxs)
        print("- idx: {}".format(idx))

        if idx>=n:
            break
        # l2.append(l.pop(i))

        complete_idxs = np.flip(np.array(list(range(1, i+1))+[idx])-1)
        complete_idxs_shift = np.roll(complete_idxs, -1)
        print("complete_idxs: {}".format(complete_idxs))
        print("complete_idxs_shift: {}".format(complete_idxs_shift))
        # complete_idxs_shift = complete_idxs[-1:]+complete_idxs[:-1]
        temp = l[idx-1]
        # for j in complete_idxs
        for j1, j2 in zip(complete_idxs[:-1], complete_idxs_shift):
            l[j1] = l[j2]
        l[0] = temp

        l2.append(l[0])
        i += 1
    # l2 = sorted(l2)
    print("l2: {}".format(l2))

    arr2 = np.array(l2)
    arr2_sort = np.sort(arr2)

    print("arr2: {}".format(arr2))
    # idx = np.where(np.diff(arr2_sort)>1)[0][0]
    # print("idx: {}".format(idx))
    # print("arr2_sort[idx]: {}".format(arr2_sort[idx]))
    # print("arr2_sort[idx+1]: {}".format(arr2_sort[idx+1]))

    # write_digraph_as_dotfile(path='collatz_1_graph_n_{}.dot'.format(n), arr_x=arr_x, arr_y=arr_y)
