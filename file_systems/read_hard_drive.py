#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import string
import sys

sys.path.append('../encryption')
import utils_encryption

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

if __name__=='__main__':
    print('Hello World!')

    f = open('/dev/loop15', 'rb')
