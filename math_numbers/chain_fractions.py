#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

import decimal
from decimal import Decimal as Dec

from PIL import Image, ImageDraw, ImageFont

decimal.getcontext().prec = 1000

def approximate_phi(n=1000):
    dec_1 = Dec(1)
    dec_x = dec_1
    dec_x_prev = dec_x

    for i in range(0, n):
        dec_x = dec_1+dec_1/dec_x
        print("\ni: {}, dec_x: {}".format(i, dec_x))
        diff = dec_x - dec_x_prev
        print("diff: {}".format(diff))

        dec_x_prev = dec_x

    return dec_x

def approx_increasing_chain_fraction(n=100):
    dec_x = Dec(n*2-1)
    # lst_dec_x = [dec_x]
    # lst_dec_x = [Dec(dec_x.to_eng_string())]
    for i in range(n-1, 0, -1):
        dec_x = Dec(i*2-1)+Dec(i*2)/dec_x
        # lst_dec_x.append(dec_x)
        # lst_dec_x.append(Dec(dec_x.to_eng_string()))

    return dec_x
    # return lst_dec_x

def approx_decreasing_chain_fraction(n=100):
    dec_x = Dec(1)
    # lst_dec_x = [dec_x]
    for i in range(1, n, 1):
        dec_x = Dec(i*2+1)+Dec(i*2)/dec_x
        # lst_dec_x.append(dec_x)
    dec_x = Dec(i*2+2)/dec_x

    return dec_x
    # return lst_dec_x

if __name__ == "__main__":
    # fibonacci number approximation
    # dec_phi = approximate_phi(n=100)

    # TODO: can be made more generic for different functions too!
    # increasing number approximation
    ds = np.array([approx_increasing_chain_fraction(n=i) for i in range(1, 100+1)])
    diffs = ds[1:]-ds[:-1]
    d1 = ds[-2]
    d2 = ds[-1]
    print("d1: {}".format(d1))
    print("d2: {}".format(d2))

    plt.close("all")

    plt.figure()

    plt.title("Plots of ds and diffs")

    plt.xlabel("n")
    plt.ylabel("y")

    p_1 = plt.plot(np.arange(0, ds.shape[0]), ds, "b.-")[0]
    p_2 = plt.plot(np.arange(0, diffs.shape[0]), diffs, "g.-")[0]

    plt.legend((p_1, p_2), ("ds", "diffs"))

    plt.show()

    diff = d1-d2
    print("diff: {}".format(diff))

    # # decreasing number approximation
    # d1 = approx_decreasing_chain_fraction(n=200)
    # d2 = approx_decreasing_chain_fraction(n=400)

    # print("d1: {}".format(d1))
    # print("d2: {}".format(d2))

    # diff = d1-d2
    # print("diff: {}".format(diff))
