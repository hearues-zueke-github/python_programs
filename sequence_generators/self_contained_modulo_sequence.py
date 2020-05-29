#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import sys

from copy import deepcopy

import numpy as np

import matplotlib.pyplot as plt

from functools import reduce
from dotmap import DotMap

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__=='__main__':
    m = 6 # modulo
    l_jump_tbls = [[i1, i2] for i1 in range(0, m) for i2 in range(0, m)]

    def get_l_a(l_jump_tbl):
        arr_jump_tbl = np.array(l_jump_tbl)
        max_n = np.sum(arr_jump_tbl)+arr_jump_tbl.shape[0]
        l = np.arange(0, max_n).tolist()
        l_a = []
        for i in l_jump_tbl:
            l_a.append(l.pop(i))
        return l_a
    

    l_equal_l_a = l_jump_tbls
    for j in range(0, 11):
        print("j: {}".format(j))
        l_new_equal_l_a = []
        for i, l_jump_tbl_1 in enumerate(l_equal_l_a, 0):
            for i, l_jump_tbl_2 in enumerate(l_jump_tbls, 0):
                l_jump_tbl = l_jump_tbl_1+l_jump_tbl_2
                l_a = (np.array(get_l_a(l_jump_tbl))%m).tolist()
                if l_a==l_jump_tbl:
                    l_new_equal_l_a.append(l_jump_tbl)
        if len(l_new_equal_l_a)==0:
            print('at j={} no more possible moves!'.format(j))
            break
        l_equal_l_a = l_new_equal_l_a

    print("l_equal_l_a: {}".format(l_equal_l_a))
