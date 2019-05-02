#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import re
import sys
import time

import itertools
import multiprocessing

import subprocess

import matplotlib.pyplot as plt

cpu_amount = multiprocessing.cpu_count()

import numpy as np

from PIL import Image

# import decimal
# prec = 50
# decimal.getcontext().prec = prec

# from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# import utils

if __name__ == "__main__":
    n = 3
    
    def get_xs_ys(n):
        xs = np.random.random((n, ))*0.5+0.1
        xs = np.cumsum(xs)
        ys = (np.random.random((n, ))-0.5)*3+2
        return xs, ys

    # plt.figure()

    # for _ in range(0, 5):
    #     xs, ys = get_xs_ys(n)
    #     plt.plot(xs, ys, ".-")

    # plt.show()



    # for _ in range(0, 5):
    #     xs, ys = get_xs_ys(n)
    #     plt.plot(xs, ys, ".-")

    plt.figure()

    xs, ys = get_xs_ys(n)
    
    plt.plot(xs, ys, ".-")

    # find a
    X = np.vstack([xs**i for i in range(0, n)]).T
    print("X:\n{}".format(X))
    Xi = np.linalg.inv(X)
    print("Xi:\n{}".format(Xi))

    asz = Xi.dot(ys)
    print("asz:\n{}".format(asz))
    
    xs2 = np.arange(xs[0], xs[-1]+0.01, 0.02)
    ys2 = np.sum([a*xs2**i for i, a in enumerate(asz, 0)], axis=0)

    plt.plot(xs2, ys2, "-")

    plt.show()
