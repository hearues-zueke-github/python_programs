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
    P_B = 21
    # Q_B = 
    
    # V_d_opt = 0.1
    V_d_opt = 0.206
    V_B = 50
    c = 4.18
    
    # c1 = P_B / Q_B
    # c2 = V_d / V_B
    
    dt = 0.1

    theta_cold = 10
    theta_start = 75
    theta_opt = 35

    theta = theta_start
    V_warm = V_d_opt*(theta_opt-theta_cold)/(theta-theta_cold)
    
    xs = [0]
    ys_theta = [theta]
    ys_V_warm = [V_warm]

    t = 0
    t_crit = 0
    theta_crit_reached = False
    for i in range(0, 1000000):
        if i % 100 == 0:
            print("i: {}".format(i))
        theta = (theta*(V_B-V_warm*dt)+theta_cold*V_warm*dt)/(V_B)+P_B*dt/(V_B*c)
        if theta > theta_opt:
            V_warm = V_d_opt*(theta_opt-theta_cold)/(theta-theta_cold)

        t += dt

        if theta_crit_reached == False and theta <= theta_opt:
            theta_crit_reached = True
            t_crit = t

        xs.append(t)
        ys_theta.append(theta)
        ys_V_warm.append(V_warm)

    # print("xs: {}".format(xs))
    # print("ys_theta: {}".format(ys_theta))

    print("t_crit: {}".format(t_crit))
    print("t_crit/60: {}".format(t_crit/60))
    print("ys_theta[-1]: {}".format(ys_theta[-1]))

    plt.figure()
    plt.title("t-theta diagram")
    plt.plot(xs, ys_theta, "b-")
    plt.plot([t_crit, t_crit], [0, theta_opt+20], "r-")
    # plt.show()

    # print("xs: {}".format(xs))
    # print("ys_theta: {}".format(ys_theta))

    plt.figure()
    plt.title("t-V_warm diagram")
    plt.plot(xs, ys_V_warm, "g-")
    plt.show()

