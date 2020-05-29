#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import sys

import matplotlib.pyplot as plt

import numpy as np

from math import factorial as fac

sys.path.append("../combinatorics/")
from different_combinations import get_all_combinations_repeat

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__=='__main__':
    m = 2
    n = 8
    arr = get_all_combinations_repeat(m, n)    
    print("arr:\n{}".format(arr))

    d = {}
    for row in arr:
        t1 = tuple(row.tolist())
        a = row.copy()
        t2 = tuple(((np.roll(a^np.hstack((a[a==0], a[a==1])), 1)+np.roll(a, -1))%2).tolist())
        d[t1] = t2

    print("d: {}".format(d))

    # first map every t to a num
    d_t_to_n = {tuple(row.tolist()): i for i, row in enumerate(arr, 0)}

    with open('binary_moving_pattern_n_{}.dot'.format(n), 'w') as f:
        f.write('digraph {\n')
        for k, v in d_t_to_n.items():
            f.write(f'  x{v}[label="{k}"];\n')
        f.write('\n')
        for k, v in d.items():
            n1 = d_t_to_n[k]
            n2 = d_t_to_n[v]
            f.write(f'  x{n1} -> x{n2};\n')

        f.write('}\n')
