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
    i = 1

    n = 500
    l = [1]
    for _ in range(0, n):
        i2 = i
        if i2%2==0:
            i2 //= 2
            if i2 in l:
                i2 = i*5+1
                if i2 in l:
                    break
        else:
            i2 = i*5+1
            if i2 in l:
                break
        i = i2
        l.append(i)

    print("l: {}".format(l))
    arr = np.array(l)
    arr_x = arr[:-1]
    arr_y = arr[1:]

    write_digraph_as_dotfile(path='collatz_1_graph_n_{}.dot'.format(n), arr_x=arr_x, arr_y=arr_y)
