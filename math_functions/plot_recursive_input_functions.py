#! /usr/bin/python2.7

import sys

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def f_1(x, i=1):
    return i + 1.0 / x

if __name__ == "__main__":
    x = np.arange(-5., 5., 0.01)

    vf1 = np.vectorize(f_1)

    def get_next_y(vf, x, y_min, y_max, i=1):
        y = vf(x)
        # print("y: {}".format(y))
        check_range = (y >= y_min) & (y <= y_max)
        # print("check_range: {}".format(check_range))
        return y[check_range]

    xs = [x.copy()]
    for i in xrange(0, 5):
        x = get_next_y(vf1, x, -5., 5., i=(i+1))
        xs.append(x.copy())

    phi = (1+5.**0.5) / 2.
    plt.figure()
    for i, x in enumerate(xs):
        plt.plot(x, np.zeros(x.shape)+i, "b.")
    plt.plot([phi, phi], [0, len(xs)], "k-")
    plt.xlabel("x")
    plt.ylabel("iteration")
    plt.show()
