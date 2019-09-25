#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
from different_combinations import get_permutation_table
import different_combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from collections import defaultdict

import numpy as np

from PIL import Image
from functools import reduce
from math import factorial
from time import time

sys.path.append("..")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
sys.path.append("../modulo_sequences/")
from utils_modulo_sequences import prettyprint_dict

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

if __name__ == "__main__":
    n_max = 3000000
    # l_2 = []
    # l_3 = []

    # i = 1
    # while i <= n_max:
    #     l_2.append(i)
    #     i *= 2

    # i = 1
    # while i <= n_max:
    #     l_3.append(i)
    #     i *= 3

    ps = list(get_primes(n_max))
    set_ps = set(ps)
    ps_part = ps[:10]

    dict_sum_three_primes = defaultdict(list)
    # dict_sum_three_primes = {}
    # for p1 in ps_part:
    #     print("p1: {}".format(p1))
    #     for p2 in ps_part:
    #         for p3 in ps_part:
    #             s = p1+p2+p3
    #             if s in set_ps:
    #                 l = dict_sum_three_primes[s]
    #                 t = (p1, p2, p3)
    #                 t = tuple(sorted(t))
    #                 if t not in l:
    #                     l.append(t)

    for p1 in ps_part:
        print("p1: {}".format(p1))
        for p2 in ps_part:
            for p3 in ps_part:
                for p4 in ps_part:
                    for p5 in ps_part:
                        s = p1+p2+p3+p4+p5
                        if s in set_ps:
                            l = dict_sum_three_primes[s]
                            t = (p1, p2, p3, p4, p5)
                            t = tuple(sorted(t))
                            if t not in l:
                                l.append(t)

    # lst_three_sum = []
    # for i1, v1 in enumerate(l_2, 0):
    #     for i2, v2 in enumerate(l_3, 0):
    #     # for i2, v2 in enumerate(l_3[i1:], i1):
    #     # for i2, v2 in enumerate(l_3[i1+1:], i1+1):
    #         for i3, v3 in enumerate(l_3, 0):
    #         # for i3, v3 in enumerate(l_3[i2:], i2):
    #         # for i3, v3 in enumerate(l_3[i2+1:], i2+1):
    #             s = v1+v2+v3
    #             if s < n_max and s in ps:
    #                 lst_three_sum.append(((i1, i2, i3), (v1, v2,v3), s))


    # n1 = 2
    # n2 = 3

    # x1 = 1
    # x2 = 1

    # l = []
    # for i in range(0, 100):
    #     print("i: {}".format(i))
    #     if x1<=x2:
    #         x1 *= n1
    #         l.append(0)
    #     else:
    #         x2 *= n2
    #         l.append(1)

    # print("l: {}".format(l))

    # calc_diff = lambda x: x[1:]-x[:-1]
    # arr = np.array(l)
