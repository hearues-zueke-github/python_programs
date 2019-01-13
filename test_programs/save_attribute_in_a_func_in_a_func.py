#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import numpy as np

def f(n):
    def g():
        pass
    g.n = n
    f.g = g
    return g

if __name__ == "__main__":
    print("Hello World!")

    g = f(1)

    print("1: g.n: {}".format(g.n))
    print("1: f.g.n: {}".format(f.g.n))
    
    print("--------")
    g1 = f(2)
    print("2: g.n: {}".format(g.n))
    print("2: g1.n: {}".format(g1.n))
    print("2: f.g.n: {}".format(f.g.n))
    
    print("--------")
    g = f(3)
    print("3: g.n: {}".format(g.n))
    print("2: g1.n: {}".format(g1.n))
    print("3: f.g.n: {}".format(f.g.n))
