#! /usr/bin/python2.7

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from math import factorial as fac

get_possible_positons = lambda n: np.sum([fac(n-1)/fac(n-1-2*i)/fac(i)**2 for i in xrange(0, (n+1)//2)])

pos_nums = np.array([get_possible_positons(i) for i in xrange(2, 21)])
print("pos_nums: {}".format(pos_nums))

diff_pos_nums = pos_nums[1:]-pos_nums[:-1]
print("diff_pos_nums: {}".format(diff_pos_nums))
