#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os

import numpy as np

from PIL import Image

import Utils

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

if __name__ == "__main__":
    path_images = HOME_DIR+"/Pictures/tiles_images/"
    file_name = "nature_1.jpeg"
    print("path_images: {}".format(path_images))
    print("file_name: {}".format(file_name))
    
    img_orig = Image.open(path_images+file_name)
    pix = np.array(img_orig)

    print("Creating some pixelated images:")
    tile_width_arr = 2**np.arange(1, 6)
    for tile_width in tile_width_arr:
        print("  tile_width: {}".format(tile_width))
        pix_crop_pixel, pix_crop_tiles_rgb = Utils.get_pixelated_pix(pix, tile_width)
        img_crop = Image.fromarray(pix_crop_pixel)

        img_crop.save(path_images+file_name.replace(".jpeg", "pxs_{}.png".format(tile_width)))
