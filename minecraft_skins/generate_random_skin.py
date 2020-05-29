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

sys.path.append("../math_numbers")
import prime_numbers_fun

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

if __name__ == '__main__':
    print('Hello World!')
    file_path_blank = PATH_ROOT_DIR+'blank.png'
    img = Image.open(file_path_blank)
    pix = np.array(img)
    idxs = np.where(pix[:, :, 0]==255)

    amount_row = idxs[0].shape[0]
    arr_rgba_random = np.random.randint(0, 255, (amount_row, 4), dtype=np.uint8)
    arr_rgba_random[:, 3] = 255

    pix[idxs] = arr_rgba_random

    img_rnd = Image.fromarray(pix)
    img_rnd.save(PATH_ROOT_DIR+'random.png')
