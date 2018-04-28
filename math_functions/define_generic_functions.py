#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from os.path import expanduser

def get_permutation_table(n):
    return np.random.permutation(np.arange(0, n))

if __name__ == "__main__":
    home_path = expanduser("~")
    print("home_path: {}".format(home_path))

    m = 15
    n = 40
    a = np.random.randint(0, m, (n, ))
    print("m: {}, n: {}".format(m, n))
    print("a: {}".format(a))

    perm_tbl = get_permutation_table(16)
    print("perm_tbl:\n{}".format(perm_tbl))
