#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

import numpy as np

from collections import namedtuple
from dotmap import DotMap

import matplotlib.pyplot as plt


def kill_half_species_approach_1(n):
    assert n%2==0

    entities = np.arange(1, n+1)

    xs = np.arange(n, 0, -1)
    rs = np.random.randint(0, n+1, (n, ))

    are_alive = rs<xs

    entities_alive = entities[are_alive]
    entities_dead = entities[~are_alive]

    entities_priority = np.hstack((entities_alive, entities_dead))
    entities_left = entities_priority[:n//2]
    entities_right = entities_priority[n//2:]

    return entities_left, entities_right


if __name__ == "__main__":
    file_name = "genetic_algor_entities_alive.pkl.gz"

    if not os.path.exists(file_name):
        alive_dict = {}
    else:
        with gzip.open(file_name, "rb") as fin:
            alive_dict = dill.load(fin)

    print("alive_dict: {}".format(alive_dict))

    ns = [10, 20, 30, 40, 50, 60, 100, 150]
    # ns = [2**i for i in range(4, 9)]

    alive_n = []
    for n in ns:
        if n in alive_dict:
            alive = alive_dict[n]
        else:
            alive = np.zeros((n, ), dtype=np.int)
            alive_dict[n] = alive
    
        alive_lst = []
        ps_lst = []
        min_alive = alive[-2]
        min_alive_max = min_alive+1000
        while True:
            entities_left, entities_right = kill_half_species_approach_1(n-2)

            alive[0] += 1
            alive[entities_left] += 1

            if np.all(alive[:-1] > min_alive):
                min_alive += 1
                if min_alive > min_alive_max:
                    break

        print("n: {}".format(n))
        print("alive: {}".format(alive))
        alive_n.append((n, alive.copy()))

    # print("alive_lst:\n{}".format(np.array(alive_lst)))

    colors = ["#FF0000",
              "#00FF00",
              "#0000FF",
              "#FF00FF",
              "#FFFF00",
              "#FF00FF",
              "#FF8888",
              "#8888FF"]
    legend_labels = list(map(str, ns))

    xs = []
    ys = []

    xs_div1 = []
    ys_div1 = []

    # for i, (n, alive) in enumerate(alive_n):
    for i, n in enumerate(ns):
        alive = alive_dict[n]
        print("\nalive: {}".format(alive))

        print("n: {}".format(n))
        print("alive.shape: {}".format(alive.shape))
        
        delta = 1/(alive.shape[0]-1)
        print("delta: {}".format(delta))
        
        x = np.arange(0, 1+delta*0.999999, delta)
        y = alive/alive[0]

        xs.append(x)
        ys.append(y)

        x_div1 = (x[:-1]+delta/2).copy()
        y_div1 = ((y[1:]-y[:-1])/delta).copy()

        xs_div1.append(x_div1)
        ys_div1.append(y_div1)

    print("")
    plt.figure()
    plt.title("Propabilities for each entities to be alive")
    plot_lst = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        print("i: {}, propability".format(i))
        p = plt.plot(x, y, "-", color=colors[i])[0]
        plot_lst.append(p)

    plt.legend(plot_lst, legend_labels)
    plt.grid()

    print("")
    plt.figure()
    plt.title("Derivation of the propabilities")
    plot_lst = []
    for i, (x, y) in enumerate(zip(xs_div1, ys_div1)):
        print("i: {}, derivation".format(i))
        p = plt.plot(x, y, "-", color=colors[i])[0]
        plot_lst.append(p)

    plt.legend(plot_lst, legend_labels)
    plt.grid()

    plt.show()

    with gzip.open(file_name, "wb") as fout:
        dill.dump(alive_dict, fout)
