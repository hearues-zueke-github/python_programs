#! /usr/bin/python3.6

import os
import pdb
import sys

import numpy as np

from PIL import Image

def apply_neighbour_logic(pix_bw, choosen_algo=0):
    height, width = pix_bw.shape[:2]

    zero_row = np.zeros((width, ), dtype=np.uint8)
    zero_col = np.zeros((height, 1), dtype=np.uint8)

    move_arr_u = lambda pix_bw: np.vstack((pix_bw[1:], zero_row))
    move_arr_d = lambda pix_bw: np.vstack((zero_row, pix_bw[:-1]))
    move_arr_l = lambda pix_bw: np.hstack((pix_bw[:, 1:], zero_col))
    move_arr_r = lambda pix_bw: np.hstack((zero_col, pix_bw[:, :-1]))

    pixs = np.zeros((5, 5, height, width), dtype=np.uint8)
    pixs[2, 2] = pix_bw

    for i in range(2, 0, -1):
        pixs[i-1, 2] = move_arr_u(pixs[i, 2])
    for i in range(2, 4):
        pixs[i+1, 2] = move_arr_d(pixs[i, 2])

    for j in range(0, 5):
        for i in range(2, 0, -1):
            pixs[j, i-1] = move_arr_l(pixs[j, i])
        for i in range(2, 4):
            pixs[j, i+1] = move_arr_r(pixs[j, i])

    var_lst = [["pix", 2, 2],

               ["pix_u", 1, 2],
               ["pix_d", 3, 2],
               ["pix_l", 2, 1],
               ["pix_r", 2, 3],

               ["pix_ul", 1, 1],
               ["pix_ur", 1, 3],
               ["pix_dl", 3, 1],
               ["pix_dr", 3, 3],

               ["pix_uu", 0, 2],
               ["pix_dd", 4, 2],
               ["pix_ll", 2, 0],
               ["pix_re", 2, 4]]

    for var_name, y, x in var_lst:
        exec("{} = pixs[{}, {}]".format(var_name, y, x))
        exec("print('var_name: {}, value: {{}}'.format({}))".format(var_name, var_name))

    fs = [
        lambda: pix_u,
        lambda: pix_d,
        lambda: pix_l,
        lambda: pix_r,
        lambda: pix_ul,
        lambda: pix_ur,
        lambda: pix_dl,
        lambda: pix_dr,
        lambda: pix_uu,
        lambda: pix_dd,
        lambda: pix_ll,
        lambda: pix_rr
        ]

    pix_bw = fs[4]().astype(np.uint8)

    return pix_bw

if __name__ == "__main__":
    pix_bw = np.random.randint(0, 2, (150, 200), dtype=np.uint8)
    apply_neighbour_logic(pix_bw)    
