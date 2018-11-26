#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import sys
import time

import numpy as np

if __name__ == "__main__":
    n = 100

    y_x_idx = np.zeros((n, n, 2), dtype=np.uint8)
    y_x_idx[:, :, 0] = np.arange(0, n).reshape((-1, 1))
    y_x_idx[:, :, 1] = np.arange(0, n).reshape((1, -1))
    y_x_idx = y_x_idx.reshape((-1, 2))


    start_1 = time.time()
    s_1_x = 0
    s_1_y = 0
    s_1_xy = 0
    for y in range(n-2, -1, -1):
        for x in range(0, n):
            s_1_x += x
            s_1_y += y
            s_1_xy += x*y
    end_1 = time.time()

    start_2 = time.time()
    s_1_x = 0
    s_1_y = 0
    s_1_xy = 0
    for y, x in y_x_idx:
        s_1_x += x
        s_1_y += y
        s_1_xy += x*y
    end_2 = time.time()

    print("time for normal two loops: {:.4}s".format(end_1-start_1))
    print("time for one loop combo:   {:.4}s".format(end_2-start_2))
