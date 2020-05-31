#! /usr/bin/python3

# -*- coding: utf-8 -*-

import datetime
import os
import sys
import subprocess
import shutil

import numpy as np

from PIL import Image

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

from utils_tetris import parse_tetris_game_data

import platform
print("platform.system(): {}".format(platform.system()))

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

if __name__ == "__main__":
    f = tempfile.NamedTemporaryFile(prefix="Test", suffix=".txt")
    TEMP_DIR = f.name[:f.name.rfind("/")]+"/"

    suffix = sys.argv[1]

    TEMP_DIR_PICS = TEMP_DIR+f"tetris_pictures_{suffix}/"
    if not os.path.exists(TEMP_DIR_PICS):
        os.makedirs(TEMP_DIR_PICS)
    else:
        shutil.rmtree(TEMP_DIR_PICS)
        os.makedirs(TEMP_DIR_PICS)

    print("TEMP_DIR_PICS: {}".format(TEMP_DIR_PICS))

    file_name = f'tetris_game_data/data_fields_{suffix}.ttrsfields'
    resize = 15
    
    DATA_FILE_PATH = PATH_ROOT_DIR+file_name

    l_colors = [
        [0x00, 0x00, 0x00],
        [0xFF, 0x00, 0x00],
        [0x00, 0xFF, 0x00],
        [0x00, 0x00, 0xFF],
        [0xFF, 0xFF, 0x00],
        [0xFF, 0x00, 0xFF],
        [0x00, 0xFF, 0xFF],
        [0x80, 0x80, 0xFF],
        [0x80, 0x80, 0x80],
    ]
    l_colors[-1:-1] = np.random.randint(64, 256, (40, 3), dtype=np.uint8).tolist()
    arr_colors = np.array(l_colors, dtype=np.uint8)


    d_data = parse_tetris_game_data(DATA_FILE_PATH)
    rows = d_data['rows']
    cols = d_data['cols']
    arr_fields = d_data['arr_fields']


    arr_1_col = np.zeros((rows, 1), dtype=np.uint8)+len(arr_colors)-1
    arr_2_col = np.zeros((rows, 5), dtype=np.uint8)+len(arr_colors)-1
    arr_1_row = np.zeros((1, cols+arr_1_col.shape[1]+arr_2_col.shape[1]), dtype=np.uint8)+len(arr_colors)-1

    for i, arr in enumerate(arr_fields, 0):
        arr = np.hstack((arr_1_col, arr, arr_2_col))
        arr = np.vstack((arr_1_row, arr, arr_1_row))

        pix = arr_colors[arr]
        img = Image.fromarray(pix)
        width, height = img.size
        img = img.resize((width*resize, height*resize))
        img.save(TEMP_DIR_PICS+f"field_nr_{i:04}.png")

    # create gif command: convert -delay 5 -loop 0 *.png animatedGIF.gif

    prefix_file_name = "output"
    extension_file_name = "mp4"
    FPS = 60
    COMMAND = f"ffmpeg -i field_nr_%04d.png -qscale 0 -r {FPS}"

    old_file_name = f"{prefix_file_name}.{extension_file_name}"
    FNULL = open(os.devnull, "wb")
    proc = subprocess.Popen(f"cd {TEMP_DIR_PICS} && {COMMAND} {old_file_name}", shell=True, stdout=FNULL)
    # proc = subprocess.Popen("cd {} && convert -delay 5 -loop 0 *.png animatedGIF.gif".format(TEMP_DIR_PICS), shell=True)
    FNULL.close()
    proc.wait()

    DIR_GIF_IMAGES = TEMP_DIR+"tetris_gifs/"
    # DIR_GIF_IMAGES = PATH_ROOT_DIR+"gifs/"
    if not os.path.exists(DIR_GIF_IMAGES):
        os.makedirs(DIR_GIF_IMAGES)

    l_hex = np.array(list('0123456789ABCDEF'))
    def get_random_hex():
        return ''.join(np.random.choice(l_hex, (4, )))


    def get_new_file_name():
        dt_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        hex_str = get_random_hex()
        return f"{extension_file_name}_suf_{suffix}_{dt_str}_{hex_str}.{extension_file_name}"

    new_file_name = get_new_file_name()

    shutil.copyfile(TEMP_DIR_PICS+old_file_name, DIR_GIF_IMAGES+new_file_name)
