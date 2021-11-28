#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9.5 -i

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

if __name__ == "__main__":
    height = 300
    width = 300

    get_new_binary_pix = lambda: np.random.randint(0, 2, (height, width, 3)).astype(np.uint8)
    
    pix = get_new_binary_pix()

    img = Image.fromarray(pix*255)
    img.save(os.path.join(TEMP_DIR, "random_image_1.png"), "PNG")
    # img.show()

    pix_1 = pix.copy()

    pix_factors = get_new_binary_pix()

    pix[:, :, 0] = np.dot(pix[:, :, 0], pix_factors[:, :, 0])
    pix[:, :, 1] = np.dot(pix[:, :, 1], pix_factors[:, :, 1])
    pix[:, :, 2] = np.dot(pix[:, :, 2], pix_factors[:, :, 2])
    
    pix %= 2
    
    img_2 = Image.fromarray(pix*255)
    img_2.save(os.path.join(TEMP_DIR, "random_image_2.png"), "PNG")
    # img_2.show()

    pix_2 = pix.copy()
