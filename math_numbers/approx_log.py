#! /usr/bin/python2.7

import decimal

import numpy as np

import matplotlib.pyplot as plt

from decimal import Decimal as Dec
from math import factorial as fac

decimal.getcontext().prec = 100

def calc_ln(x, iterations):
    s = Dec("0")
    for j in xrange(0, iterations):
        i = j*2+1
        s += 1/Dec("{}".format(i))*((x-Dec("1"))/(x+Dec("1")))**i

    return Dec("2")*s

def calc_ln_part(x):
    s = Dec("0")
    d_2 = Dec("2")
    x = Dec(x)
    print("in generator x: {}".format(x))
    j = 0

    while True:
        i = j*2+1
        s += (1/Dec(i)*((x-Dec("1"))/(x+Dec("1")))**i)*d_2
        j += 1
        yield s

def calc_ln_precision(x, precision):
    precision_prev = decimal.getcontext().prec
    decimal.getcontext().prec = precision+50
    gen = calc_ln_part(x)
    y_prev = str(next(gen))
    print("y_prev: {}".format(y_prev))
    for i in xrange(1, 100):
        y_dec = next(gen)
        y = str(y_dec)
        # print("i: {}".format(i))
        print("y: {}".format(y))

        if len(y_prev) > precision:
            is_ok = True
            for i, (a, b) in enumerate(zip(y_prev, y)):
                if a != b:
                    is_ok = False
                    break
                if i > precision:
                    break

            if is_ok:
                break

        y_prev = y

    decimal.getcontext().prec = precision
    y_dec = Dec(str(y_dec))
    decimal.getcontext().prec = precision_prev

    return y_dec

if __name__ == "__main__":
    a = Dec("0.2334")
    print("a: {}".format(a))


    x = Dec("3")
    ln_x = calc_ln(x, 10)
    
    i = 0
    for s in calc_ln_part(x):
        # print("i: {}, s: {}".format(i, s))
        i += 1
        if i > 100:
            break

    z = calc_ln_precision(x, 80)

    print("z: {}".format(z))

    print("x: {}".format(x))
    print("ln_x: {}".format(ln_x))
    