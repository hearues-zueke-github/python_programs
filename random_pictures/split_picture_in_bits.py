#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import colorsys
import datetime
import dill
import gzip
import os
import shutil
import string
import subprocess
import sys
import time
import traceback

import numpy as np

import matplotlib.pyplot as plt

from dotmap import DotMap

from indexed import IndexedOrderedDict
from collections import OrderedDict
from dotmap import DotMap
from PIL import Image, ImageDraw, ImageFont

import utils

if __name__ == "__main__":
    main_folder = "images/random_images_split/"

    utils.check_create_folders(main_folder)

    width = 400
    height = 300
    pix = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    pixs_bits = np.empty((3, 8, height, width), dtype=np.uint8)

    print("pix.shape: {}".format(pix.shape))

    random_imag_folder = main_folder+"random_image_1/"
    utils.check_create_folders(random_imag_folder)

    Image.fromarray(pix).save(random_imag_folder+"random.png")

    # split in each channel
    for i, channel_name in enumerate(["r", "g", "b"]):
        arr_ch = pix[:, :, i]

        Image.fromarray(arr_ch).save(random_imag_folder+"channel_only_{}.png".format(channel_name))

        # split in each bit
        for bit in range(0, 8):
            arr_bit = ((arr_ch>>(7-bit))&0x01)
            pixs_bits[i, bit] = arr_bit
            arr_bit = arr_bit*255

            Image.fromarray(arr_bit).save(random_imag_folder+"channel_{}_{}_bit_{}.png".format(i, channel_name, bit))
