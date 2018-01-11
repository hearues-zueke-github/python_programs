#! /usr/bin/python3.5

import decimal
import math

from decimal import Decimal as D

import matplotlib.pyplot as plt

decimal.getcontext().prec = 300

def get_taylor_approximation(x, l):
    """
        @param x: number in Decimal!
        @param l: a list of all values at position 0, l = [f(0), df/dx(0), d^2f/dx^2(0),...]
    """
    s = l[0]
    f = D("1")
    for i, k in enumerate(l[1:]):
        j = D("{}".format(i+1))
        f *= j
        s += k*x**j/f
        # print("\ni: {}".format(i))
        # print("j: {}".format(j))
        # print("f: {}".format(f))    
    return s

def approx_e():
    n = 500
    l = [D("1") for i in range(0, n)]
    xs = []
    ys = []
    for i in range(0, 301):
        for j in range(0, 10):
            x = D("{}.{}".format(i, j))
            y = get_taylor_approximation(x, l)
            xs.append(float(x))
            ys.append(float(y))

    plt.figure()

    plt.yscale("log")

    plt.plot(xs, ys, "b-")
    plt.plot(xs, [math.exp(x) for x in xs], "g-")

    plt.show()

def approx_sin():
    xs = []
    for i in range(-18, 1):
        for j in range(0, 1000):
            x = D("{}.{:03d}".format(i, 999-j))
            xs.append(x)
    xs.append(D("0"))
    for i in range(0, 18+1):
        for j in range(0, 1000):
            x = D("{}.{:03d}".format(i, j))
            xs.append(x)
    print("xs: {}".format(xs))
    def get_ys(n):
        l = [D("{}".format((lambda i: 0 if i % 2 == 0 else 1 if i % 4 == 1 else -1)(i))) for i in range(0, n)]
        ys = []
        for x in xs:
            y = get_taylor_approximation(x, l)
            ys.append(float(y))
        return ys

    yss = []
    for n in range(2, 50, 10):
        yss.append(get_ys(n))

    plt.figure()

    xs = [float(x) for x in xs]
    print("xs: {}".format(xs))

    plt.plot(xs, [math.sin(x) for x in xs], "-")
    for ys in yss:
        plt.plot(xs, [float(y) for y in ys], "-")

    plt.ylim([-1.3, 1.3])

    plt.show()

approx_sin()
