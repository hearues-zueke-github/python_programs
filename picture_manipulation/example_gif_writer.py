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

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

# from . import create_lambda_functions

import create_lambda_functions

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all

import numpy as np
from array2gif import write_gif

dataset = np.array([
    [
        [[255, 0, 0], [255, 0, 0]],  # red intensities
        [[0, 255, 0], [0, 255, 0]],  # green intensities
        [[0, 0, 255], [0, 0, 255]]   # blue intensities
    ], [
        [[0, 0, 255], [0, 0, 255]],
        [[0, 255, 0], [0, 255, 0]],
        [[255, 0, 0], [255, 0, 0]]
    ]
])
write_gif(dataset, 'rgbbgr.gif', fps=5)

# or for just a still GIF
write_gif(dataset[0], 'rgb.gif')
