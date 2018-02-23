#! /usr/bin/python2.7

import numpy as np

import matplotlib.pyplot as plt

from math import factorial as fac

def get_amount_combinations(n, shift):
    return np.sum([fac(n-1)//fac(n-1-shift-2*i)//fac(shift+i)//fac(i) for i in xrange(0, (n+1-shift)//2)])

comb_shift_0 = [get_amount_combinations(n, 0) for n in xrange(2, 11)]
print("comb_shift_0: {}".format(comb_shift_0))
comb_shift_1 = [get_amount_combinations(n, 1) for n in xrange(2, 11)]
print("comb_shift_1: {}".format(comb_shift_1))
comb_shift_2 = [get_amount_combinations(n, 2) for n in xrange(3, 11)]
print("comb_shift_2: {}".format(comb_shift_2))
