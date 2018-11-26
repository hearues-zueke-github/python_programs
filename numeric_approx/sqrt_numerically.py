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
    a = Dec(136)
    print("a: {}".format(a))

    x = Dec(1)
    xs = [x]
    k = 6
    dec_k = Dec(k)

    for i in range(0, 30):
        x = (x*(k-1)+a/(x**(k-1)))/dec_k
        xs.append(x)

    for (i1, x1), (i2, x2) in zip(enumerate(xs[1:], 1), enumerate(xs[:-1], 0)):
        print("\ni1: {}, x1: {}".format(i1, x1))
        diff = x2-x1
        print("diff: {}".format(diff))
