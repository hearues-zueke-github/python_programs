#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import re
import string

import numpy as np

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

from config_file import FILE_NAME_JPG

if __name__ == '__main__':
    DIR_IMAGES = PATH_ROOT_DIR+'images/'
    if not os.path.exists(DIR_IMAGES):
        os.makedirs(DIR_IMAGES)

    FILE_PATH_ORIG = DIR_IMAGES+FILE_NAME_JPG

    FILE_NAME_PNG = re.sub(r'\.jpg$', '', FILE_NAME_JPG)+'.png'
    FILE_PATH_PNG = DIR_IMAGES+FILE_NAME_PNG

    img = Image.open(FILE_PATH_ORIG)
    if not os.path.exists(FILE_PATH_PNG):
        img.save(FILE_PATH_PNG)

    pix = np.array(img)
    print("pix.shape: {}".format(pix.shape))

    def create_hist_256(u, c):
        assert u.dtype == np.dtype('uint8')
        c_new = np.zeros((256, ), dtype=c.dtype)
        c_new[u] = c
        return c_new

    l_c = [create_hist_256(*np.unique(pix[..., i], return_counts=True)) for i in range(0, 3)]

    xs = np.arange(0, 256)
    ys_r = l_c[0]
    ys_g = l_c[1]
    ys_b = l_c[2]

    fig, axs = plt.subplots(nrows=3, ncols=1)
    axs[0].bar(xs, ys_r, color='#FF0000', width=1.)
    axs[1].bar(xs, ys_g, color='#00FF00', width=1.)
    axs[2].bar(xs, ys_b, color='#0000FF', width=1.)
    plt.show(block=False)

    # create simple rgb reduces images!
    for bit in range(1, 8):
        amount = 2**bit
        
        arr_idx = np.arange(0, amount + 1) * (256 / amount)
        arr_idx_round = np.round(arr_idx).astype(np.int)

        arr_val = np.arange(0, amount) * (255 / (amount - 1))
        arr_val_round = np.round(arr_val).astype(np.int)
        
        print("bit: {}\n- arr_idx_round: {}\n- arr_val_round: {}".format(bit, arr_idx_round, arr_val_round))

        arr_map = np.zeros((256, ), dtype=np.uint8)
        for i1, i2, v in zip(arr_idx_round[:-1], arr_idx_round[1:], arr_val):
            arr_map[i1:i2] = v
        print("arr_map: {}".format(arr_map))

        pix2 = np.empty(pix.shape, dtype=np.uint8)
        pix2[..., 0] = arr_map[pix[..., 0]]
        pix2[..., 1] = arr_map[pix[..., 1]]
        pix2[..., 2] = arr_map[pix[..., 2]]
        img2 = Image.fromarray(pix2)
        img2.save(DIR_IMAGES+FILE_NAME_PNG.replace('.png', '_c{}b{}.png'.format(pix2.shape[2], bit)))
