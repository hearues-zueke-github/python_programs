#! /usr/bin/python

# -*- coding: utf-8 -*-

import numpy as np

if __name__ == "__main__":
    print("Hello World!")

    V1 = 1.5 # e.g. Fanta
    V2 = 1.0 # e.g. Coca-Cola

    # what is the ratios of deltas so that the concentration of Fanta and Coca-Cola is 1:1?
    # or in other words: 50% Fanta, 50% Coca-Cola

    n = 9
    deltas = [1./n for _ in range(0, n)]
    # deltas = [1/3., 1/3., 1/3.]

    change_val = 0.030984566
    deltas[-2] += change_val
    deltas[-1] -= change_val

    print("np.sum(deltas): {}".format(np.sum(deltas)))

    # assert V2 == np.sum(deltas)

    p = 0 # 0% Coca-Cola concentration

    def calc_new_p(V, delta, p):
        return (delta*1+(V-delta)*p)/V

    print("0: p: {}".format(p))
    for i, delta in enumerate(deltas, 1):
        p = calc_new_p(V1, delta, p)
        print("{}: p: {}".format(i, p))
