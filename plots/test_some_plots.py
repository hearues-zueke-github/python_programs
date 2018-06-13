#! /usr/bin/python3.5

import numpy as np

import matplotlib.pyplot as plt

def plot_1():
    plt.figure()

    xs = [0, 5, 5, 7, 8, 3]
    ys = [3, 3, 6, 8, 4, 5]

    plt.plot(xs, ys, "b-")

    plt.show()

def plot_2():
    plt.figure()
    xs = np.random.randint(0, 4, (1000, ))
    ys = np.random.randint(0, 4, (1000, ))
    plt.plot(xs, ys, "b-")
    plt.plot(xs[0], ys[0], "k.")
    plt.show()

def plot_all_possible_lines():
    plt.figure()
    n = 6
    # for i in range(0, n):
    #     plt.plot(np.arange(0, n), np.zeros((n, ))+i, "b-")
    #     plt.plot(np.zeros((n, ))+i, np.arange(0, n), "b-")

    for x1 in range(0, n):
        for y1 in range(0, n):
            for x2 in range(0, n):
                for y2 in range(0, n):
                    if x1 == x2 and y1 == y2:
                        continue
                    plt.plot([x1, x2], [y1, y2], "b-", linewidth=0.1)


    # TODO: make this plot more efficent + make another plot where there
    # are only the cross points of all lines!
    plt.show()

def plot_3():
    plt.figure()
    n = 20
    plt.plot([0, n, n, 0, 0], [0, 0, n, n, 0], "k-")
    # xs = np.arange(0, n+1, 1)
    # ys = np.arange(n, -1, -1)
    
    for i in range(1, n):
        plt.plot([0, i], [i, n], "b-")
    # for i in range(1, n):
        plt.plot([i, n], [n, n-i], "g-")
        plt.plot([n, n-i], [n-i, 0], "r-")
        plt.plot([n-i, 0], [0, i], "c-")
    plt.show()

# plot_1()
# plot_2()
# plot_all_possible_lines()
plot_3()
