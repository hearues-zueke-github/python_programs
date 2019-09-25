#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports

import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import traceback

# Needed for excel tabels
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles.borders import Border, Side, BORDER_THIN, BORDER_MEDIUM
from openpyxl.styles import Alignment, borders, Font

import matplotlib.pyplot as plt

import decimal
from decimal import Decimal as Dec
precision = 50
decimal.getcontext().prec = precision

from PIL import Image

# Needed for DataFrame (databases etc.)
import numpy as np
import pandas as pd
import pandas.io.sql as sql

from dotmap import DotMap
from functools import reduce
from math import factorial

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

def dec_cos(d, n_max=14):
    d_1 = Dec(1)

    s = Dec(1)
    n = 2
    sig = -1
    while n <= n_max:
        s += d**n/factorial(n)*sig
        n += 2
        sig *= -1

    return s

if __name__ == '__main__':
    print("Hello World!")

    # solve integral from -2 to 2 for (x**3*cos(x/2)+1/2)*sqrt(4-x**2)
    def f(x):
        # print("- x: {}".format(x))
        part1 = x**3*dec_cos(x/Dec(2))+Dec(1)/Dec(2)
        part2 = np.sqrt(Dec(4)-x**2)
        return part1*part2
        # return (x**3*dec_cos(x/Dec(2))+Dec(1)/Dec(2))*np.sqrt(Dec(4)-x**2)

    a = Dec(-2)
    b = Dec(2)

    diff = b-a
    n = 100000
    delta_x = diff/n

    prev_f = f(a)

    s = Dec(0)
    # using trapeze integral approx.
    for i in range(1, n+1):
        now_f = f(a+delta_x*i)
        s += delta_x*(prev_f+now_f)/2
        prev_f = now_f
        # print("i: {}, s: {}".format(i, s))
    
    print("n: {}".format(n))
    print("delta_x: {}".format(delta_x))
    print("s: {}".format(s))
