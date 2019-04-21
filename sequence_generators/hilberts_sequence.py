#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import os
import sys
import time

import numpy as np

from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

import utils_sequence


def get_hilbert_curve_sequence(n):
    fr = np.array([0, 1])
    ff = np.array([0, 0])
    rf = np.array([1, 0])
    ll = np.array([-1, -1])

    arr = np.array([1, 1])
    if n == 1:
        return arr

    for i in range(1, n):
        if i % 2 == 1:
            arr = np.hstack((arr*-1, rf, arr, ll, arr, fr, arr*-1))
        else:
            arr = np.hstack((arr*-1, fr, arr, ff, arr, rf, arr*-1))

    return arr


def get_hilberts_curve_field(n):
    dir_str = {-1: 'L', 0: 'F', 1: 'R'}
    get_lst_str = lambda lst: list(map(lambda x: dir_str[x], lst))

    new_dir = {
        ('R', 'L'): (np.array([-1, 0]), 'U'),
        ('D', 'L'): (np.array([0, 1]), 'R'),
        ('L', 'L'): (np.array([1, 0]), 'D'),
        ('U', 'L'): (np.array([0, -1]), 'L'),

        ('U', 'F'): (np.array([-1, 0]), 'U'),
        ('R', 'F'): (np.array([0, 1]), 'R'),
        ('D', 'F'): (np.array([1, 0]), 'D'),
        ('L', 'F'): (np.array([0, -1]), 'L'),

        ('L', 'R'): (np.array([-1, 0]), 'U'),
        ('U', 'R'): (np.array([0, 1]), 'R'),
        ('R', 'R'): (np.array([1, 0]), 'D'),
        ('D', 'R'): (np.array([0, -1]), 'L'),
    }

    # n = 9
    l_side = 2**n
    arr = get_hilbert_curve_sequence(n)
    p = np.array([l_side-1, 0])
    ps = np.zeros((l_side**2, 2), dtype=np.int)
    ds = np.zeros((l_side**2, 2), dtype=np.int)

    ps[0] = p

    lst = np.array(get_lst_str(arr.tolist()))

    field = np.zeros((l_side, l_side), dtype=np.int)

    idx = 1
    if n % 2 == 0:
        p += [0, 1]
        ps[1] = p
        d_now = 'R'
        ds[0] = [0, 1]

    else:
        p += [-1, 0]
        ps[1] = p
        d_now = 'U'
        ds[0] = [-1, 0]
    field[p[0], p[1]] = idx
    idx += 1

    for i, d in enumerate(lst):
        p_rel, d_now = new_dir[(d_now, d)]
        p += p_rel
        ps[i+2] = p
        ds[i+1] = p_rel
        field[p[0], p[1]] = idx
        idx += 1

    print("arr:\n{}".format(arr))
    print("lst:\n{}".format(lst))
    print("field:\n{}".format(field))

    return field, ps, ds


if __name__ == "__main__":
    n = 4

    field, ps, ds = get_hilberts_curve_field(n)

    print("ps:\n{}".format(ps))
    print("ds:\n{}".format(ds))

    # colors = utils_sequence.all_possibilities_changing_one_position(64, 3)*4
    
    # col_jump = 8000
    # colors = colors[np.arange(0, colors.shape[0], col_jump)]
    # print("colors.shape[0]: {}".format(colors.shape[0]))
    colors = np.zeros((510, 3), dtype=np.uint8)
    colors[:256, 0] = np.arange(0, 256)
    colors[:256, 1] = np.arange(0, 256)
    colors[256:, 0] = np.arange(1, 255)[::-1]
    colors[256:, 1] = np.arange(1, 255)[::-1]

    l_side = 2**n
    needed_colors = l_side**2
    if needed_colors > colors.shape[0]:
        colors_first = colors.copy()
        while needed_colors > colors.shape[0]:
            colors = np.vstack((colors, colors_first))
    colors = colors[:needed_colors]

    pix = colors[field.reshape((-1, ))].reshape((l_side, l_side, -1)).astype(np.uint8)
    res_factor = 1
    img = Image.fromarray(pix).resize((l_side*res_factor, l_side*res_factor))
    # img.show()
    path_images = "images/hilberts_curves/"
    if not os.path.exists(path_images):
        os.makedirs(path_images)
    img.save(path_images+"hilberts_curve_n_{}.png".format(n))
    # img.save(path_images+"hilberts_curve_n_{}_col_jump_{}.png".format(n, col_jump))


    # Dots with spaces!
    jump = 4

    l_side_jump = l_side+(l_side-1)*(jump-1)
    pix2 = np.zeros((l_side_jump, l_side_jump, 3), dtype=np.uint8)
    ps2 = (ps*jump).T
    ds2 = ds.T

    cols = [
        (0x00, 0xFF, 0xFF),
        (0xFF, 0xFF, 0x00),
        (0xFF, 0x00, 0xFF),
        (0xFF, 0x00, 0x00),
    ]

    for i, it in enumerate(range(jump-1, -1, -1), 0):
        pix2[:] = 0
        for j in range(0, 2):
            y, x = ps2+ds2*((i+j)%jump)
            pix2[y, x] = cols[0]
        # for j in range(0, jump):
        #     y, x = ps2+ds2*j
        #     pix2[y, x] = cols[0]
            # pix2[y, x] = cols[(j+it)%jump]
        # pix2[ps2[0, -1], ps2[1, -1]] = cols[(it) % jump]
        if i > 2:
            pix2[ps2[0, -1], ps2[1, -1]] = cols[0]
        else:
            pix2[ps2[0, -1], ps2[1, -1]] = (0, 0, 0)

        img2 = Image.fromarray(pix2)
        resize_factor = 3
        img2 = img2.resize((l_side_jump*resize_factor, l_side_jump*resize_factor))
        img2.save(path_images+"colorful_path_i_{}.png".format(i))
