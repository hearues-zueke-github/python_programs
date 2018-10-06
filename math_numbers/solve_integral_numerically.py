#! /usr/bin/python3.6

import os
import pdb

import numpy as np

import decimal
from decimal import Decimal as Dec

if __name__ == "__main__":
    # solve the integral from -2 to 2 for the function:
    # f(x) = (x**3*cos(x/2)+1/2)*sqrt(4-x**2)

    decimal.getcontext().prec = 100

    def own_cos(x):
        # d = Dec(1)
        d = 1
        s = Dec(1)
        for i in range(2, 30, 2):
            d *= (i-1)*i
            s += x**i/d
        return s

    def integral_approx(f, a, b, n):
        # first get the x's
        diff = b-a
        dx = diff/n
        # xs = np.hstack((a+np.arange(0, n)*dx, [b]))
        # print("diff: {}".format(diff))
        # print("dx: {}".format(dx))
        # print("xs: {}".format(xs))

        s = 0
        y_prev = f(a)
        for i in range(1, n+1):
            y_new = f(a+dx*i)
            s += (y_prev+y_new)/2*dx
            y_prev = y_new

        return s
    
    f = lambda x: (x**3*own_cos(x/Dec(2))+Dec(1)/Dec(2))*np.sqrt(Dec(4)-x**2)
    s_1 = integral_approx(f, Dec(-2), Dec(2), 300000)
    print("s_1: {}".format(s_1))
