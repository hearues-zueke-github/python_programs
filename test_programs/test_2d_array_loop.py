#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import sys
import time

import numpy as np

if __name__ == "__main__":
    n = 200

    start_1 = time.time()
    s_1_x = 0
    s_1_y = 0
    s_1_xy = 0
    # for y in range(n-1, -1, -1):
    for y in range(0, n):
        for x in range(0, n):
            s_1_x += x
            s_1_y += y
            s_1_xy += x*y
    end_1 = time.time()

    start_2 = time.time()
    y_x_idx = np.zeros((n, n, 2), dtype=np.int)
    y_x_idx[:, :, 0] = np.arange(0, n).reshape((-1, 1))
    y_x_idx[:, :, 1] = np.arange(0, n).reshape((1, -1))
    y_x_idx = y_x_idx.reshape((-1, 2))
    s_2_x = 0
    s_2_y = 0
    s_2_xy = 0
    for y, x in y_x_idx:
        s_2_x += x
        s_2_y += y
        s_2_xy += x*y
    end_2 = time.time()

    start_3 = time.time()
    y_x_idx_lst = y_x_idx.tolist()
    s_3_x = 0
    s_3_y = 0
    s_3_xy = 0
    for y, x in y_x_idx_lst:
        s_3_x += x
        s_3_y += y
        s_3_xy += x*y
    end_3 = time.time()

    start_3_1 = time.time()
    y_x_idx_lst = y_x_idx.tolist()
    s_3_1_x = 0
    s_3_1_y = 0
    s_3_1_xy = 0
    for t in y_x_idx_lst:
        x = t[0]
        y = t[1]
        s_3_1_x += x
        s_3_1_y += y
        s_3_1_xy += x*y
    end_3_1 = time.time()

    start_4 = time.time()
    y_x_idx = np.zeros((n, n, 2), dtype=np.int)
    y_x_idx[:, :, 0] = np.arange(0, n).reshape((-1, 1))
    y_x_idx[:, :, 1] = np.arange(0, n).reshape((1, -1))
    y_x_idx = y_x_idx.reshape((-1, 2))
    s_4_x = np.sum(y_x_idx[:, 0])
    s_4_y = np.sum(y_x_idx[:, 1])
    s_4_xy = np.sum(np.multiply.reduce(y_x_idx, axis=1))
    # for y, x in y_x_idx:
    #     s_4_x += x
    #     s_4_y += y
    #     s_4_xy += x*y
    end_4 = time.time()

    print("time for normal two loops:      {:.4}s".format(end_1-start_1))
    print("time for one loop combo:        {:.4}s".format(end_2-start_2))
    print("time for one loop combo (list): {:.4}s".format(end_3-start_3))
    print("time for one loop combo (one var): {:.4}s".format(end_3_1-start_3_1))
    print("time for numpy functions:       {:.4}s".format(end_4-start_4))

    print("s_1_x: {}".format(s_1_x))
    print("s_1_y: {}".format(s_1_y))
    print("s_1_xy: {}".format(s_1_xy))

    print("s_2_x: {}".format(s_2_x))
    print("s_2_y: {}".format(s_2_y))
    print("s_2_xy: {}".format(s_2_xy))

    print("s_3_x: {}".format(s_3_x))
    print("s_3_y: {}".format(s_3_y))
    print("s_3_xy: {}".format(s_3_xy))

    print("s_3_1_x: {}".format(s_3_1_x))
    print("s_3_1_y: {}".format(s_3_1_y))
    print("s_3_1_xy: {}".format(s_3_1_xy))

    print("s_4_x: {}".format(s_4_x))
    print("s_4_y: {}".format(s_4_y))
    print("s_4_xy: {}".format(s_4_xy))
