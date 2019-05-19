#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import mmap
import os
import re
import sys
import time

import itertools
import multiprocessing

from multiprocessing import Process, Pipe
# from multiprocessing import shared_memory # in python3.8 available!

import matplotlib
import matplotlib.pyplot as plt

import subprocess

import numpy as np

from PIL import Image

# import decimal
# prec = 50
# decimal.getcontext().prec = prec

# from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__ == "__main__":
    n = 6
    xs = np.cumsum(np.random.random((n, ))*2+0.1)+1
    print("xs: {}".format(xs))

    ys = np.random.random((n, ))*6+0.1
    print("ys: {}".format(ys))

    asz_factors = []
    lst_quadratic_points = []
    # calculate a, b, c factors for three points in between!
    for i in range(0, n-2):
        xs_part = xs[i:i+3]
        ys_part = ys[i:i+3]
        print("xs_part: {}".format(xs_part))
        print("ys_part: {}".format(ys_part))

        A = np.tile(xs_part, 3).reshape((3, 3)).T**np.arange(2, -1, -1)
        A_inv = np.linalg.inv(A)
        asz = A_inv.dot(ys_part)
        print("A: {}".format(A))
        print("A_inv: {}".format(A_inv))

        asz_factors.append(asz)

        x_start, x_end = xs_part[0], xs_part[-1]
        xs_quadratic = np.arange(x_start, x_end, (x_end-x_start)/100)
        print("xs_quadratic: {}".format(xs_quadratic))
        ys_1 = np.tile(xs_quadratic, 3).reshape((3, -1))**np.arange(2, -1, -1).reshape((-1, 1))
        ys_quadratic = asz.dot(ys_1)
        # ys_quadratic = np.sum(ys_2, axis=0)
        print("ys_1: {}".format(ys_1))
        print("ys_quadratic: {}".format(ys_quadratic))

        lst_quadratic_points.append((xs_quadratic, ys_quadratic))

    lst_quadratic_linear_points = []
    for i in range(0, n-3):
        asz1 = asz_factors[i]
        asz2 = asz_factors[i+1]

        x_start, x_end = xs[i+1], xs[i+2]
        xsz = np.arange(x_start, x_end, (x_end-x_start)/100)
        xs_quadratic = np.tile(xsz, 3).reshape((3, -1))**np.arange(2, -1, -1).reshape((-1, 1))
        ys_quadratic_1 = asz1.dot(xs_quadratic)
        ys_quadratic_2 = asz2.dot(xs_quadratic)

        alphas = np.arange(0, 100)/100
        ys_quadratic = ys_quadratic_1*(1-alphas)+ys_quadratic_2*alphas

        lst_quadratic_linear_points.append((xsz, ys_quadratic))


    plt.figure()
    plt.title("Do quadratic linear interpolation!")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xs, ys, "b.")
    plt.plot(xs, ys, "g-")

    for xs_quadratic, ys_quadratic in lst_quadratic_points:
        plt.plot(xs_quadratic, ys_quadratic, "r-")

    for xs_, ys_ in lst_quadratic_linear_points:
        plt.plot(xs_, ys_, "k-")

    plt.show()
