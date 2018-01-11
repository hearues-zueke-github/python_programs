#! /usr/bin/python3.5

import decimal
import math

import numpy as np

import matplotlib.pyplot as plt

from decimal import Decimal as D

decimal.getcontext().prec = 2000

pi = D("3.1415926535897932384626433832795028841071693993751058209749445923078164062862089986280348253421170679")

def calc_unknown_series():
    str_zero = "0"
    str_one = "1"

    l = []
    for i in range(1, 200):
        l.append(D("0."+str_zero*i+str_one*i))

    s = np.sum(l)
    print("s: {}".format(s))

def calc_pi():
    p = D("1")
    for i in range(1, 1000, 2):
        p *= (D(i+1)*D(i+1))/(D(i)*D(i+2))

    print("p: {}".format(p))

calc_pi()

print("pi/D(4): {}".format(pi/D(2)))
