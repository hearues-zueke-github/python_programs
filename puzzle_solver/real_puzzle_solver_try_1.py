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
# from multiprocessing import shared_memory # in python3.8 available!

import matplotlib
import matplotlib.pyplot as plt

import subprocess

import numpy as np

from PIL import Image

# import decimal
# prec = 50
# decimal.getcontext().prec = prec

# from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__ == "__main__":
    img = Image.open('images/puzzels_part_2.png')
    # img.show()

    pix = np.array(img)
    pix = pix[..., :3]
    print("pix.shape: {}".format(pix.shape))

    # img2 = Image.fromarray(pix[:, :, :3])

    # img.show()
    # img2.save('images/puzzels_part_2_no_alpha.png')

    pix_gray = np.dot(pix, [0.299, 0.587, 0.144])

    pix_gray_uint8 = pix_gray.astype(np.uint8)

    dx = pix_gray[:, 1:]-pix_gray[:, :-1]
    dy = pix_gray[1:]-pix_gray[:-1]

    dx_uint8 = (dx+128).astype(np.uint8)
    dy_uint8 = (dy+128).astype(np.uint8)

    img_dx = Image.fromarray(dx_uint8)
    img_dy = Image.fromarray(dy_uint8)

    # img_dx.show()
    # img_dy.show()

    dg = np.sqrt(dx[:-1]**2+dy[:, :-1]**2)
    dg_norm = (dg-np.min(dg))/(np.max(dg)-np.min(dg))*255.
    img_dg = Image.fromarray(dg_norm.astype(np.uint8))
    # img_dg.show()

    img_dx.save('images/puzzels_part_2_dx.png')
    img_dy.save('images/puzzels_part_2_dy.png')
    img_dg.save('images/puzzels_part_2_dg.png')

    for i in range(5, 55, 5):
        print("i: {}".format(i))
        dg_threshold = (((dg>i)+0)*255).astype(np.uint8)
        Image.fromarray(dg_threshold).save('images/puzzels_part_2_threshold_{:03}.png'.format(i))
    # dg_30 = (((dg>20)+0)*255).astype(np.uint8)
    # dg_30 = (((dg>20)+0)*255).astype(np.uint8)
    # dg_30 = (((dg>20)+0)*255).astype(np.uint8)
