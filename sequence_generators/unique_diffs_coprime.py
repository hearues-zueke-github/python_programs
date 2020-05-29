#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

# TODO: need to be python convension confirm! (coding standard as much as possible!)

import datetime
import dill
import gzip
import os
import pdb
import shutil
import string
import sys

from copy import deepcopy

import matplotlib.pyplot as plt

import numpy as np

from dotmap import DotMap

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization

def gcd(a, b):
    if a==b:
        return a
    if a<b:
        a, b = b, a
    while b>0:
        t = a%b
        a = b
        b = t
    return a


# def is_int_sqrt(n):
#     x_prev = n
#     x_now = (n//1+1)//2
#     while x_now<x_prev:
#         t = (n//x_now+x_now)//2
#         x_prev = x_now
#         x_now = t
#     return x_now**2==n


# def int_sqrt(n):
#     x_prev = n
#     x_now = (n//1+1)//2
#     while x_now<x_prev:
#         t = (n//x_now+x_now)//2
#         x_prev = x_now
#         x_now = t
#     return x_now


if __name__ == "__main__":
    print('Hello World!')
    l = [1]
    s_diffs = set()
    l_not_used = []
    n_last = 2
    amount_new_vals = 20000
    for i in range(0, amount_new_vals):
        if i%100==0:
            print("i: {}".format(i))
        last_v = l[-1]
        # print('------')
        # print("l: {}".format(l))
        # print("last_v: {}".format(last_v))
        # print("s_diffs: {}".format(s_diffs))
        # print("l_not_used: {}".format(l_not_used))
        # print("n_last: {}".format(n_last))
        is_found_new_val = False
        for j, k in enumerate(l_not_used, 0):
            diff_now = abs(k-last_v)
            if gcd(k, last_v)!=1 or diff_now in s_diffs:
                continue

            is_found_new_val = True
            # print("j: {}".format(j))
            l.append(k)
            s_diffs.add(diff_now)
            l_not_used.pop(j)
            break

        if is_found_new_val:
            continue

        while True:
            diff_now = abs(n_last-last_v)
            if gcd(n_last, last_v)!=1 or diff_now in s_diffs:
                l_not_used.append(n_last)
                n_last += 1
                continue

            l.append(n_last)
            s_diffs.add(diff_now)
            n_last += 1
            break

    print("l: {}".format(l))

    plt.figure()

    plt.title('Unique diffs and coprime sequence with len = {}'.format(amount_new_vals))
    plt.plot(np.arange(0, len(l)), l, '.b', markersize=5)

    plt.show(block=False)
