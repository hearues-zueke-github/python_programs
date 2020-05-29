#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os

import numpy as np

from PIL import Image

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

import platform
print("platform.system(): {}".format(platform.system()))

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

if __name__ == "__main__":
    f = tempfile.NamedTemporaryFile(prefix="Test", suffix=".txt")
    temp_dir = f.name[:f.name.rfind("/")]+"/"

    TEMP_DIR_PICS = temp_dir+"tetris_pictures/"
    if not os.path.exists(TEMP_DIR_PICS):
        os.makedirs(TEMP_DIR_PICS)

    print("TEMP_DIR_PICS: {}".format(TEMP_DIR_PICS))

    # f = tempfile.NamedTemporaryFile(prefix="Test", suffix=".txt", delete=False, )
    DATA_FILE_PATH = PATH_ROOT_DIR+'tetris_game_data/data_fields_test.ttrsfields'

    with open(DATA_FILE_PATH, "rb") as f:
        l_bytes = list(f.read())

    arr_colors = np.array([
        [0x00, 0x00, 0x00],
        [0xFF, 0x00, 0x00],
        [0x00, 0xFF, 0x00],
        [0x00, 0x00, 0xFF],
        [0xFF, 0xFF, 0x00],
        [0xFF, 0x00, 0xFF],
        [0x00, 0xFF, 0xFF],
        [0x80, 0x80, 0xFF],
        [0x80, 0x80, 0x80],
    ], dtype=np.uint8)

    rows, cols = l_bytes[:2]
    field_cells = rows*cols

    resize = 15

    l_rest = l_bytes[2:]

    arr_1_col = np.zeros((rows, 1), dtype=np.uint8)+8
    arr_1_row = np.zeros((1, cols+2), dtype=np.uint8)+8

    amount_of_fields = len(l_rest)//rows//cols
    for i in range(0, amount_of_fields):
        l = l_rest[field_cells*i:field_cells*(1+i)]
        arr = np.array(l).reshape((rows, cols))

        arr = np.hstack((arr_1_col, arr, arr_1_col))
        arr = np.vstack((arr_1_row, arr, arr_1_row))

        pix = arr_colors[arr]
        img = Image.fromarray(pix)
        img = img.resize((cols*resize, rows*resize))
        img.save(TEMP_DIR_PICS+"field_nr_{:03}.png".format(i))
