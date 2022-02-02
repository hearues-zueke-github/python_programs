#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import cv2
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

IMAGES_PATH = os.path.join(TEMP_DIR, 'random_images_1920x1080')
mkdirs(IMAGES_PATH)

def convert_images_to_video():
    video_name = os.path.join(IMAGES_PATH, 'video.avi')

    images = sorted([img for img in os.listdir(IMAGES_PATH) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(IMAGES_PATH, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(filename=video_name, fourcc=fourcc, fps=60, frameSize=(width, height))

    for img_idx, image in enumerate(images, 1):
        print(f"img_idx: {img_idx}")
        video.write(cv2.imread(os.path.join(IMAGES_PATH, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    # convert_images_to_video()
    # sys.exit()

    w = 1920//2
    h = 1080//2

    idx_img = 0
    print(f"idx_img: {idx_img}")

    pix = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    pix_buffer = pix.copy()
    
    pix_buffer[:] = pix
    # pix_buffer[0, 0:0%60] = (0, 0, 0)
    # pix_buffer[0:5, 60:65] = (128, 128, 128)
    img = Image.fromarray(pix_buffer)
    img = img.resize(size=(w*2, h*2), resample=Image.NEAREST)
    # img.show()

    img.save(os.path.join(IMAGES_PATH, f'img_{idx_img:06}.png'))
    idx_img += 1

    arr_idx = np.arange(0, w*h*3, dtype=np.uint64)
    amount = w * h * 3 // 5

    for i in range(1, 60*60):
        print(f"idx_img: {idx_img}")
        arr_idx[:] = np.random.permutation(arr_idx)
        add_count = np.random.randint(1, 31)
        pix[arr_idx[:amount] // 3 // w, (arr_idx[:amount] // 3) % w, arr_idx[:amount] % 3] += add_count
        pix_buffer[:] = pix
        # pix_buffer[0, 0:i%60] = (0, 0, 0)
        # pix_buffer[0:5, 60:65] = (128, 128, 128)
        img = Image.fromarray(pix_buffer)
        img = img.resize(size=(w*2, h*2), resample=Image.NEAREST)

        img.save(os.path.join(IMAGES_PATH, f'img_{idx_img:06}.png'))
        idx_img += 1

    convert_images_to_video()
    # sys.exit()
