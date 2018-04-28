#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from os.path import expanduser

def get_2d_idx_table(m, n):
    idx_2d_table = np.zeros((m, n, 2)).astype(np.int)

    idx_2d_table[:, :, 0] += np.arange(0, m).reshape((m, -1))
    idx_2d_table[:, :, 1] += np.arange(0, n).reshape((-1, n))

    return idx_2d_table

if __name__ == "__main__":
    home_path = expanduser("~")
    print("home_path: {}".format(home_path))

    idx_frame_3x3 = get_2d_idx_table(3, 3)
    print("idx_frame_3x3:\n{}".format(idx_frame_3x3))

    h = 5
    w = 5

    val_table = np.random.randint(0, 10, (h, w))
    # print("val_table:\n{}".format(val_table))

    val_table_ext_row = np.vstack((val_table[0],
                                   val_table[:],
                                   val_table[-1]))
    # print("val_table_ext_row:\n{}".format(val_table_ext_row))
    
    val_table_ext = np.hstack((val_table_ext_row[:, 0].reshape((-1, 1)),
                               val_table_ext_row[:],
                               val_table_ext_row[:, -1].reshape((-1, 1))))
    print("val_table_ext:\n{}".format(val_table_ext))

    j = 1
    i = 2

    m = 3
    n = 3
    
    idx_y_table = np.random.randint(0, h, (m*n, ))
    idx_x_table = np.random.randint(0, w, (m*n, ))

    print("idx_y_table:\n{}".format(idx_y_table))
    print("idx_x_table:\n{}".format(idx_x_table))

    idx_table = np.vstack((idx_y_table, idx_x_table))

    print("idx_table:\n{}".format(idx_table))

    new_vals = val_table[idx_table.tolist()].reshape((m, n))
    print("new_vals:\n{}".format(new_vals))
