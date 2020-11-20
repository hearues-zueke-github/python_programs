#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import traceback

import numpy as np
import pandas as pd

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile

from PIL import Image

sys.path.append('..')
from utils import mkdirs
from utils_all import int_sqrt

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

if __name__ == '__main__':
    # first read the file as a bytearray!
    file_name = 'test_file.abc'
    file_dir_path = '/tmp/'
    file_path = file_dir_path + file_name

    path_dir_img_encoded = file_dir_path + 'img_encoded/'
    mkdirs(path_dir_img_encoded)

    with open(file_path, 'rb') as f:
        bs_file_content = f.read()
        arr_content = np.array(list(bs_file_content), dtype=np.uint8)

    length = arr_content.shape[0]

    len_nonce = 8
    arr_constant = (np.arange(0, len_nonce) + 1).astype(np.uint8)[::-1]
    arr_nonce = np.random.randint(0, 256, (len_nonce, ), dtype=np.uint8)

    # 1 byte ... file_name length
    # len(file_name) ... space for file_name
    # 8 byte ... file orig length
    # length ... space for file content
    len_file_name = len(file_name)
    assert len_file_name < 256
    length_total = 1 + len_file_name + len_nonce + 8 + length

    # only for grayscale images right now!
    height = int_sqrt(length_total)
    width = height
    while height * width < length_total:
        width += 1

    length_image = width * height

    arr_image_content = np.hstack((
        arr_content,
        np.random.randint(0, 256, (length_image - length_total, ), dtype=np.uint8),
    ))

    # TODO: do the simple encryption part here!


    bs_encode_content = bytes(arr_image_content)

    bs_length = (length).to_bytes(8, byteorder='big')
    bs = (
        bytes([len_file_name]) +
        file_name.encode("utf-8") +
        bytes(arr_nonce) +
        bs_length +
        bs_encode_content
    )

    assert len(bs) == height * width

    arr = np.array(list(bs), dtype=np.uint8)
    pix = arr.reshape((height, width))
    img = Image.fromarray(pix)

    img.save(path_dir_img_encoded + 'encoded.png')
