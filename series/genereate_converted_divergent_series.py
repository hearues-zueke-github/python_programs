#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import sys

import numpy as np

import matplotlib.pyplot as plt

import decimal
from decimal import Decimal as Dec

decimal.getcontext().prec = 1000

if __name__ == "__main__":
    sums = []
    prods = []
    sums_1 = []

    s = Dec(0)
    p = Dec(1)
    s_1 = Dec(0)
    
    j = Dec(0)
    dec_1 = Dec(1)
    for i in range(0, 1000):
        j += 1

        s += dec_1/j
        p *= dec_1/s
        s_1 += p
    
        sums.append(s)
        prods.append(p)
        sums_1.append(s_1)


    for (i1, (s1, p1, s11)), (i2, (s2, p2, s12)) in \
    zip(
        enumerate(zip(sums[:-1], prods[:-1], sums_1[:-1]), 1),
        enumerate(zip(sums[1:], prods[1:], sums_1[1:]), 2)):
        print("\ni1: {}, i2: {}".format(i1, i2))
        # print("    s1: {}".format(s1))
        # print("    p1: {}".format(p1))
        print("    s11: {}".format(s11))
        
        # print("    diff_s: {}".format(s2-s1))
        # print("    diff_p: {}".format(p2-p1))
        print("    diff_p1: {}".format(s12-s11))
