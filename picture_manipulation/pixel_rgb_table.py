#! /usr/bin/python3.6

import os
import sys

import numpy as np

from PIL import Image

def get_all_rbgs():
    rgbs = np.zeros((16, 16, 256, 256, 3), dtype=np.uint8)

    values_2d = np.zeros((256, 256), dtype=np.uint8)
    values_2d[:] = np.arange(0, 256)

    rgbs[:, :, :, :, 2] = values_2d
    rgbs[:, :, :, :, 1] = values_2d.T

    for y in range(0, 16):
        for x in range(0, 16):
            rgbs[y, x, :, :, 0] = y*16+x

    return rgbs.reshape((-1, 3))

def get_new_rgbs_table(rgbs):
    rgbs_sum = np.sum(rgbs.astype(np.int)*256**np.arange(2, -1, -1), axis=1)
    rgbs_sum = rgbs_sum >> 1
    # return (lambda x: np.vstack((x//256**2, x//256**1, x//256**0)))(rgbs_sum//2)
    return ((rgbs_sum.reshape((-1, 1))//256**np.arange(2, -1, -1)) % 256).astype(np.uint8)

if __name__ == "__main__":
    rgbs = get_all_rbgs()

    get_list_rgb_tuples = lambda rgbs: rgbs.view("u1,u1,u1").reshape((-1, )).tolist()

    rgbs_new = get_new_rgbs_table(rgbs)
