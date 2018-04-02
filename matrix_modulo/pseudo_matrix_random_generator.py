#! /usr/bin/python2.7

import pylab
import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def show_matrix(A, n):
    x = -0.2
    y = 0.1

    fig, ax = plt.subplots(figsize=(12, 9))
    res = ax.imshow(pylab.array(A), cmap=plt.cm.jet, interpolation='nearest')
    for i, line in enumerate(A):
        for j, val in enumerate(line):
            # if val > 0:
            plt.text(j+x, i+y, "{}".format(val), fontsize=8)
            # pass

    plt.title("Matrix A, n: {}".format(n), y=1.08)
    plt.xlabel("x")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.ylabel("y")

    ax.xaxis.set_major_locator(ticker.FixedLocator((xrange(0, n))))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((xrange(0, n))))

    ax.yaxis.set_major_locator(ticker.FixedLocator((xrange(0, n))))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter((xrange(0, n))))

    plt.subplots_adjust(bottom=0.1, top=0.85) #left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    cb = fig.colorbar(res)
    plt.show()

if __name__ == "__main__":
    n = 12
    A = np.random.randint(0, 10, (n, n))

    show_matrix(A, n)
