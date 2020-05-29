#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

# TODO: need to be python convension confirm! (coding standard as much as possible!)

import datetime
import dill
import gzip
import inspect
import os
import pdb
import shutil
import string
import sys

import numpy as np
from array2gif import write_gif

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

# from . import create_lambda_functions

import create_lambda_functions

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all

if __name__ == '__main__':
    print('Hello World!')

    kwargs = {'dtype': np.uint8}
    f = np.array([[2, 3, 4, 5], [1, 4, 6, 7], [8, 0, 3, 1]], **kwargs)
    print("f.shape: {}".format(f.shape))
    print("f:\n{}".format(f))

    y, x = f.shape

    ft = 2
    # append f with ft
    ff = np.empty((y+ft*2, x+ft*2), **kwargs)
    ff[:] = 0

    arrs = np.empty((ft*2+1, ft*2+1), dtype=object)

    for j in range(0, ft*2+1):
        for i in range(0, ft*2+1):
            arrs[j, i] = ff[j:y+j, i:x+i]
    
    p = arrs[ft, ft]
    u = arrs[ft-1, ft]
    dl = arrs[ft+1, ft-1]

    def exchange_f(ft, ff, f):
        y, x = f.shape
        ff[ft:ft+y, ft:ft+x] = f
        ff[ft:ft+y, :ft] = f[:, -ft:]
        ff[ft:ft+y, -ft:] = f[:, :ft]
        ff[:ft, :] = ff[y:y+ft, :]
        ff[-ft:, :] = ff[ft:ft+ft, :]

    exchange_f(ft, ff, f)

    print("ff:\n{}".format(ff))
    print("p:\n{}".format(p))
    print("u:\n{}".format(u))
    print("dl:\n{}".format(dl))

    print("Change f completely!")
    f = np.array([[1, 2, 3, 4], [5, 6, 3, 4], [0, 9, 8, 7]], **kwargs)
    exchange_f(ft, ff, f)

    print("f:\n{}".format(f))
    print("ff:\n{}".format(ff))
    print("p:\n{}".format(p))
    print("u:\n{}".format(u))
    print("dl:\n{}".format(dl))
