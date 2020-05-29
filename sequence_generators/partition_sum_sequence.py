#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import datetime
import dill
import gzip
import inspect
import os
import pdb
import shutil
import string
import sys

from copy import deepcopy

import matplotlib.pyplot as plt

import numpy as np
from array2gif import write_gif

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization

if __name__ == "__main__":
    n = 100
    l = [0]*(n-1)

    s1 = 0
    for i in range(1, n):
        s1 += i
        if s1>=n:
            break
        l[s1-1] += 1
        s2 = s1
        i1 = i
        for i2 in range(i+1, n):
            s2 = s2-i1+i2
            if s2>=n:
                break
            l[s2-1] += 1
            i1 = i2
    print("l: {}".format(l))
