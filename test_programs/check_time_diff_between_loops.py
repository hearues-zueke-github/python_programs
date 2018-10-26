#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import sys
import time

import numpy as np

if __name__ == "__main__":
    n = 200

    y_x_idx = np.zeros((n, n, 2), dtype=np.int)
    y_x_idx[:, :, 0] = np.arange(0, n).reshape((-1, 1))
    y_x_idx[:, :, 1] = np.arange(0, n).reshape((1, -1))
    y_x_idx = y_x_idx.reshape((-1, 2))


    start_1 = time.time()
    s_1_x = 0
    s_1_y = 0
    s_1_xy = 0
    for _ in range(0, 20):
     for y in range(0, n):
        for x in range(0, n):
            s_1_x += x
            s_1_y += y
            s_1_xy += x*y
    end_1 = time.time()

    ys, xs = y_x_idx.T

    start_2 = time.time()
    s_2_x = 0
    s_2_y = 0
    s_2_xy = 0
    for _ in range(0, 20):
     for y, x in zip(ys, xs):
     # for y, x in y_x_idx:
        s_2_x += x
        s_2_y += y
        s_2_xy += x*y
    end_2 = time.time()

    print("time for normal two loops: {:.4}s".format(end_1-start_1))
    print("time for one loop combo:   {:.4}s".format(end_2-start_2))

    print("s_1_x: {}, s_1_y: {}, s_1_xy: {}".format(s_1_x, s_1_y, s_1_xy))
    print("s_2_x: {}, s_2_y: {}, s_2_xy: {}".format(s_2_x, s_2_y, s_2_xy))
