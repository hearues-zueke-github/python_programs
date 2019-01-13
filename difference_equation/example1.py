#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import sys
import time

import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # c1 = 1.16/209
    P_B = 50
    Q_B = 209
    
    V_d = 0.206
    V_B = 50
    
    c1 = P_B / Q_B
    c2 = V_d / V_B
    
    dt = 0.1
    theta1 = 10

    theta = 70

    xs = [0]
    ys = [theta]

    t = 0
    for _ in range(0, 30000):
        theta = theta + (c1+(theta1-theta)*c2)*dt
        t += dt
        xs.append(t)
        ys.append(theta)

    print("xs: {}".format(xs))
    print("ys: {}".format(ys))

    plt.figure()
    plt.plot(xs, ys)
    plt.show()
