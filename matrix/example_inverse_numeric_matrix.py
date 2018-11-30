#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import numpy as np

if __name__ == "__main__":
    print("Hello World!")

    x = 3

    arr = np.array([[-1,0,x],[0,1,x],[x,x,1]])

    print("arr: {}".format(arr))

    print("np.linalg.inv(arr): {}".format(np.linalg.inv(arr)))

    d = 1-x+x**2

    inv_numeric = np.array([[-1+x, -x**2, x], [-x**2, x**2, -x], [x, -x, 1]])/d

    print("inv_numeric: {}".format(inv_numeric))
