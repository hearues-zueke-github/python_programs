#! /usr/bin/python3

import time
import datetime
import dill
import os
import subprocess
import sys

import matplotlib.pyplot as plt

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__=='__main__':
    print('Hello World!')

    def f(n):
        if n<1:
            return 0

        n1 = n//4
        n2 = (3*n)//4

        if n1 in f.a:
            x1 = f.a[n1]
        else:
            x1 = f(n1)
            f.a[n1] = x1

        if n2 in f.a:
            x2 = f.a[n2]
        else:
            x2 = f(n2)
            f.a[n2] = x2

        return x1+x2+n
    f.a = {}

    xs = []
    ys = []

    for n in range(1, 1000001):
        if n%1000==0:
            print("n: {}".format(n))
        y = f(n)
        # print("n: {}, f(n): {}".format(n, y))
        xs.append(n)
        ys.append(y)

    plt.figure()

    plt.plot(xs, ys, 'b-o', markersize=2)

    plt.show()
    # plt.show(block=False)
