#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import operator

import numpy as np

from copy import deepcopy

import sys
sys.path.append("../encryption")

import Utils

if __name__ == "__main__":
    n = 1000000 # amount of elements = modulo = number base

    lines = 10
    factors = np.random.randint(1, n, (lines, 2)).astype(object)

    x = np.random.randint(0, n)
    print("x: {}".format(x))
    xs = [x]

    def apply_linear_function(x, n, factors):
        for a0, a1 in factors:
            x = (a0+a1*x) % n
        return int(x)

    while True:
        x_new = apply_linear_function(x, n, factors)
        if x_new in xs:
            break
        x = x_new
        xs.append(x_new)

    print("xs: {}".format(xs))
