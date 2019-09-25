#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import mmap
import os
import re
import sys
import time

import itertools
import multiprocessing

from multiprocessing import Process, Pipe

import matplotlib
import matplotlib.pyplot as plt

import subprocess

import numpy as np

from PIL import Image

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__ == "__main__":
    # take an image
    
    file_name = 'autumn-colorful-colourful-33109.jpg'
    # file_name = '12507539_791729790953056_7511920104969898811_n.jpg'
    file_name_changed = file_name.replace('.jpg', '_lower_amount_colors.png')
    img = Image.open(ROOT_PATH+'images/'+file_name)
    pix = np.array(img)
    h, w, channels = pix.shape

    colors = pix.reshape((-1, 3))
    colors_orig = colors.copy()

    rgb_sum = np.sum(colors.astype(np.uint64)*256**np.arange(2, -1, -1).astype(np.uint64), axis=1)
    idx_argsort = np.argsort(rgb_sum)

    rows = 5000
    idxs_rows = idx_argsort[:idx_argsort.shape[0]-idx_argsort.shape[0]%rows].reshape((-1, rows)).T

    print("idxs_rows: {}".format(idxs_rows))
    for idxs_row in idxs_rows[1:]:
        colors[idxs_row] = colors[idxs_rows[0]].copy()

    changed_colors = np.sum(np.any(colors!=colors_orig, axis=1))
    print("amount changed_colors: {}".format(changed_colors))

    pix_new = colors.reshape((h, w, channels))

    print("pix.shape: {}".format(pix.shape))
    print("colors.shape: {}".format(colors.shape))

    img_new = Image.fromarray(pix_new)
    img_new.save(ROOT_PATH+'images/'+file_name_changed)
