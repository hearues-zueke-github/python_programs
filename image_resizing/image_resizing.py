#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import re
import sys
import time

import itertools
import multiprocessing

import subprocess

cpu_amount = multiprocessing.cpu_count()

import numpy as np

from PIL import Image

# import decimal
# prec = 50
# decimal.getcontext().prec = prec

# from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# import utils

if __name__ == "__main__":
    pix = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(pix)
    # img.show()

    # choose a frame
    x1 = 30
    x2 = 130
    y1 = 50
    y2 = 110

    c = (0, 0, 0)
    pix[y1:y2, x1] = c
    pix[y1:y2, x2] = c
    pix[y1, x1:x2] = c
    pix[y2, x1:x2] = c

    img2 = Image.fromarray(pix)
    # img2.show()

    pix2 = pix[y1:y2+1, x1:x2+1]
    img3 = Image.fromarray(pix2)
    img3.show()
