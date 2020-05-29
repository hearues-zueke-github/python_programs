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

import platform
print("platform.system(): {}".format(platform.system()))

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

if __name__ == "__main__":
    f = tempfile.NamedTemporaryFile(prefix="Test", suffix=".txt")
    temp_dir = f.name[:f.name.rfind("/")]+"/"

    suffix = sys.argv[1]
    # suffix = "5"

    TEMP_DIR_PICS = temp_dir+f"tetris_pictures_{suffix}/"
    if not os.path.exists(TEMP_DIR_PICS):
        os.makedirs(TEMP_DIR_PICS)
    else:
        shutil.rmtree(TEMP_DIR_PICS)
        os.makedirs(TEMP_DIR_PICS)

    print("TEMP_DIR_PICS: {}".format(TEMP_DIR_PICS))

    file_name = f'tetris_game_data/data_fields_{suffix}.ttrsfields'
    resize = 15
    
    DATA_FILE_PATH = PATH_ROOT_DIR+file_name

    with open(DATA_FILE_PATH, "rb") as f:
        l_bytes = list(f.read())

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

    rows, cols = l_bytes[:2]
    field_cells = rows*cols


    l_rest = l_bytes[2:]

    arr_1_col = np.zeros((rows, 1), dtype=np.uint8)+len(arr_colors)-1
    arr_1_row = np.zeros((1, cols+2), dtype=np.uint8)+len(arr_colors)-1

    amount_of_fields = len(l_rest)//rows//cols
    for i in range(0, amount_of_fields):
        l = l_rest[field_cells*i:field_cells*(1+i)]
        arr = np.array(l).reshape((rows, cols))

        arr = np.hstack((arr_1_col, arr, arr_1_col))
        arr = np.vstack((arr_1_row, arr, arr_1_row))

        pix = arr_colors[arr]
        img = Image.fromarray(pix)
        img = img.resize((cols*resize, rows*resize))
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

    DIR_GIF_IMAGES = PATH_ROOT_DIR+"gifs/"
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
