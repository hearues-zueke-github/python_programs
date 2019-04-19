#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import decimal
import math

from decimal import Decimal as Dec

import matplotlib.pyplot as plt

decimal.getcontext().prec = 300

s = Dec("3.")
x = Dec("1.")
x_prev = x
dec_2 = Dec("2.")

for i in range(0, 10):
    x_prev = x
    x = (x**2+s) / (dec_2*x)
    print("i: {}".format(i))
    print("x: {}".format(x))
    diff = x-x_prev
    print("diff: {}".format(diff))
