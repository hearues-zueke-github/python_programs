#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9 -i

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
import multiprocessing as mp

from collections import defaultdict
from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from hashlib import sha256
from io import BytesIO
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)


def create_random_images_linear_interpolation():
    images_path = '/media/doublepmcl/new_exfat/random_images/'

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    h, w = 1080//2, 1920//2
    hr, wr = 1080, 1920

    n = 100
    parts = 5 # linear interpolations
    pixses = np.array([np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(0, 100)])
    print("pixses.shape: {}".format(pixses.shape))
    pixses = np.vstack((pixses, pixses[:2].copy()))
    for i in range(0, n):
        print("i: {}".format(i))
        pix1 = pixses[i].astype(np.double)
        pix2 = pixses[i+1].astype(np.double)
        for j in range(0, parts):
            pix = (pix1*(parts-j)/parts+pix2*j/parts).astype(np.uint8)
            img = Image.fromarray(pix).resize((wr, hr), Image.LANCZOS)
            img.save(images_path+"img_{:04}.png".format(i*parts+j))


def create_images_with_quadratic_linear_interpolation():
    images_path = '/media/doublepmcl/new_exfat/images_quadratic_interpolation/'

    images_name = ['nature_1.jpg', 'nature_2.jpg', 'nature_3.jpg', 'nature_4.jpg', 'nature_5.jpg']

    min_width = 10000
    pixses = []
    for image_name in images_name:
        img = Image.open(images_path+image_name)
        # img = Image.open(images_path+'nature_1.jpg')

        w_orig, h_orig = img.width, img.height
        ratio = w_orig/h_orig
        img = img.resize((int(480*ratio), 480), Image.LANCZOS)
        # img.show()

        pix = np.array(img)
        print("pix.shape: {}".format(pix.shape))
        if min_width > pix.shape[1]:
            min_width = pix.shape[1]

        pixses.append(pix)

    print("min_width: {}".format(min_width))

    # crop all images to the same width!
    for i in range(0, len(pixses)):
        pixses[i] = pixses[i][:, :min_width].copy()

    pixses = np.array(pixses)

    # for pix in pixses:
    #     Image.fromarray(pix).show()

    current_image_num = 0
    picture_name_template = 'new_image_{:04}.png'

    pixses = pixses.astype(np.double)
    # first, interpolate 3 images for the left quadratic part!
    pix1, pix2, pix3 = pixses[:3]

    h, w, chans = pix1.shape

    xs = np.arange(0, 3)
    X = np.tile(xs, 3).reshape((3, 3)).T**np.arange(2, -1, -1)
    X_inv = np.linalg.inv(X)
    # print("xs: {}".format(xs))
    # print("X: {}".format(X))
    # print("X_inv: {}".format(X_inv))

    asz_left = np.dstack((pix1, pix2, pix3)).reshape((h, w, 3, 3)).transpose(0, 1, 3, 2).dot(X_inv.T)
    print("asz_left.shape: {}".format(asz_left.shape))

    xs_inter = np.arange(0, 1, 0.1)
    # xs_inter = [1.1, 1.5, 1.8, 2.0]
    for x in xs_inter:
        print("x: {}".format(x))
        pix_new_1 = asz_left.dot(np.array((x, x, 1))**np.arange(2, -1, -1))
        pix_new_1[pix_new_1 < 0.] = 0.
        pix_new_1[pix_new_1 > 255.] = 255.
        pix_new_1 = pix_new_1.astype(np.uint8)
        img_new_1 = Image.fromarray(pix_new_1)
        img_new_1.save(images_path+picture_name_template.format(current_image_num))
        print("current_image_num: {}".format(current_image_num))
        current_image_num += 1
        # img_new_1.show()

    for i in range(1, len(pixses)-2):
        pix1, pix2, pix3 = pixses[i:i+3]

        asz_right = np.dstack((pix1, pix2, pix3)).reshape((h, w, 3, 3)).transpose(0, 1, 3, 2).dot(X_inv.T)
        print("asz_right.shape: {}".format(asz_right.shape))

        for x in xs_inter:
            print("x: {}".format(x))
            pix_calc_1 = asz_left.dot(np.array((x+1, x+1, 1))**np.arange(2, -1, -1))
            pix_calc_2 = asz_right.dot(np.array((x, x, 1))**np.arange(2, -1, -1))
            pix_new_1 = pix_calc_1*(1-x)+pix_calc_2*x
            # pix_new_1 = pix_calc_2
            pix_new_1[pix_new_1 < 0.] = 0.
            pix_new_1[pix_new_1 > 255.] = 255.
            pix_new_1 = pix_new_1.astype(np.uint8)
            img_new_1 = Image.fromarray(pix_new_1)
            img_new_1.save(images_path+picture_name_template.format(current_image_num))
            print("current_image_num: {}".format(current_image_num))
            current_image_num += 1

        asz_left = asz_right
            # img_new_1.show()

    for x in xs_inter+1:
        print("x: {}".format(x))
        pix_new_1 = asz_left.dot(np.array((x, x, 1))**np.arange(2, -1, -1))
        pix_new_1[pix_new_1 < 0.] = 0.
        pix_new_1[pix_new_1 > 255.] = 255.
        pix_new_1 = pix_new_1.astype(np.uint8)
        img_new_1 = Image.fromarray(pix_new_1)
        img_new_1.save(images_path+picture_name_template.format(current_image_num))
        print("current_image_num: {}".format(current_image_num))
        current_image_num += 1


# TODO: create operations in between with xor, sort x, sort y, etc.
# TODO: create a interesting sequence of pictures!
def create_random_images_with_quadratic_linear_interpolation():
    images_path = '/media/doublepmcl/new_exfat/random_images_2/'
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # images_name = ['nature_1.jpg', 'nature_2.jpg', 'nature_3.jpg', 'nature_4.jpg', 'nature_5.jpg']



    # min_width = 10000
    # pixses = []
    # for image_name in images_name:
    #     img = Image.open(images_path+image_name)
    #     # img = Image.open(images_path+'nature_1.jpg')

    #     w_orig, h_orig = img.width, img.height
    #     ratio = w_orig/h_orig
    #     img = img.resize((int(480*ratio), 480), Image.LANCZOS)
    #     # img.show()

    #     pix = np.array(img)
    #     print("pix.shape: {}".format(pix.shape))
    #     if min_width > pix.shape[1]:
    #         min_width = pix.shape[1]

    #     pixses.append(pix)

    # print("min_width: {}".format(min_width))

    # crop all images to the same width!
    # for i in range(0, len(pixses)):
    #     pixses[i] = pixses[i][:, :min_width].copy()

    # pixses = np.array(pixses)

    pixses = np.random.randint(0, 256, (120, 540, 960, 3), dtype=np.uint8)
    pixses = np.vstack((pixses, pixses[:1].copy()))

    # for pix in pixses:
    #     Image.fromarray(pix).show()

    current_image_num = 0
    picture_name_template = 'new_image_{:04}.png'

    pixses = pixses.astype(np.double)
    # first, interpolate 3 images for the left quadratic part!
    pix1, pix2, pix3 = pixses[:3]

    h, w, chans = pix1.shape

    xs = np.arange(0, 3)
    X = np.tile(xs, 3).reshape((3, 3)).T**np.arange(2, -1, -1)
    X_inv = np.linalg.inv(X)
    # print("xs: {}".format(xs))
    # print("X: {}".format(X))
    # print("X_inv: {}".format(X_inv))

    asz_left = np.dstack((pix1, pix2, pix3)).reshape((h, w, 3, 3)).transpose(0, 1, 3, 2).dot(X_inv.T)
    print("asz_left.shape: {}".format(asz_left.shape))

    xs_inter = np.arange(0, 1, 1/30) # 0.1)
    # xs_inter = [1.1, 1.5, 1.8, 2.0]
    for x in xs_inter:
        print("x: {}".format(x))
        pix_new_1 = asz_left.dot(np.array((x, x, 1))**np.arange(2, -1, -1))
        pix_new_1[pix_new_1 < 0.] = 0.
        pix_new_1[pix_new_1 > 255.] = 255.
        pix_new_1 = pix_new_1.astype(np.uint8)
        img_new_1 = Image.fromarray(pix_new_1).resize((1920, 1080), Image.LANCZOS)
        img_new_1.save(images_path+picture_name_template.format(current_image_num))
        print("current_image_num: {}".format(current_image_num))
        current_image_num += 1
        # img_new_1.show()

    for i in range(1, len(pixses)-2):
        pix1, pix2, pix3 = pixses[i:i+3]

        asz_right = np.dstack((pix1, pix2, pix3)).reshape((h, w, 3, 3)).transpose(0, 1, 3, 2).dot(X_inv.T)
        print("asz_right.shape: {}".format(asz_right.shape))

        for x in xs_inter:
            print("x: {}".format(x))
            pix_calc_1 = asz_left.dot(np.array((x+1, x+1, 1))**np.arange(2, -1, -1))
            pix_calc_2 = asz_right.dot(np.array((x, x, 1))**np.arange(2, -1, -1))
            pix_new_1 = pix_calc_1*(1-x)+pix_calc_2*x
            # pix_new_1 = pix_calc_2
            pix_new_1[pix_new_1 < 0.] = 0.
            pix_new_1[pix_new_1 > 255.] = 255.
            pix_new_1 = pix_new_1.astype(np.uint8)
            img_new_1 = Image.fromarray(pix_new_1).resize((1920, 1080), Image.LANCZOS)
            img_new_1.save(images_path+picture_name_template.format(current_image_num))
            print("current_image_num: {}".format(current_image_num))
            current_image_num += 1

        asz_left = asz_right
            # img_new_1.show()

    for x in xs_inter+1:
        print("x: {}".format(x))
        pix_new_1 = asz_left.dot(np.array((x, x, 1))**np.arange(2, -1, -1))
        pix_new_1[pix_new_1 < 0.] = 0.
        pix_new_1[pix_new_1 > 255.] = 255.
        pix_new_1 = pix_new_1.astype(np.uint8)
        img_new_1 = Image.fromarray(pix_new_1).resize((1920, 1080), Image.LANCZOS)
        img_new_1.save(images_path+picture_name_template.format(current_image_num))
        print("current_image_num: {}".format(current_image_num))
        current_image_num += 1


if __name__ == "__main__":
    # random path where my HDD is!
    # create_random_images_linear_interpolation()    
    
    # create_images_with_quadratic_linear_interpolation()
    create_random_images_with_quadratic_linear_interpolation()
