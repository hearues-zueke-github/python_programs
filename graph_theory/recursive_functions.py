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
    l = [0, 0, 1]
    for i in range(3, 101):
        i2 = l[i-1]-l[i-2]
        print("i2: {}".format(i2))
        if i2<0:
            break
        v = l[i-1]+l[i-2]-l[i2]-l[i-3]
        print("v: {}".format(v))
        l.append(v)

    print("l: {}".format(l))
