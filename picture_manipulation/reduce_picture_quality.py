#! /usr/bin/python3.6

import dill
import os
import pdb
import sys

import numpy as np

from dotmap import DotMap
from PIL import Image

def save_pictures_split(pix, img_dir, x_split=2, y_split=2):
    height, width, _ = pix.shape

    print("height: {}".format(height))
    print("width: {}".format(width))

    height_mod = (height//y_split)*y_split
    width_mod = (width//x_split)*x_split

    print("height_mod: {}".format(height_mod))
    print("width_mod: {}".format(width_mod))

    for y in range(0, y_split):
        pix_y_split = pix[np.arange(y, height_mod, y_split)]
        # globals()["pix_y_split"] = pix_y_split
        # sys.exit(-1)
        for x in range(0, x_split):
            pix_yx_split = pix_y_split[:, np.arange(x, width_mod, x_split)]

            Image.fromarray(pix_yx_split).save(img_dir+"split_y_{}_x_{}.png".format(y, x))
            print("y: {}, x: {}".format(y, x))

def save_picture_reduces_pixel_quality(pix, img_dir, used_steps=128):
    byte_map = np.arange(0, used_steps)*(256//(used_steps-1))
    byte_map[-1] = 255
    byte_map = byte_map.astype(np.uint8)
    globals()["byte_map"] = byte_map

    print("used_steps: {}".format(used_steps))
    print("byte_map: {}".format(byte_map))

    argmin = np.argmin(np.abs(pix.reshape(pix.shape+(1, )).astype(np.int)-byte_map), axis=-1)
    globals()["argmin"] = argmin
    pix_reduced = byte_map[argmin]
    globals()["pix_reduced"] = pix_reduced

    Image.fromarray(pix_reduced).save("reduced_bytes_used_steps_{:03}.png".format(used_steps))


if __name__ == "__main__":
    # file_name = "pexels-photo-236047.jpeg"
    file_name = "fall-autumn-red-season_resized.jpg"
    path_image = "images/{}".format(file_name)

    assert os.path.exists(path_image)
    path_dir = "images/split_images_{}/".format(file_name.replace(".jpg", ""))
    # path_dir = "images/split_images_{}/".format(file_name.replace(".jpeg", ""))

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    img = Image.open(path_image)
    pix = np.array(img)
    # save_pictures_split(pix, path_dir, x_split=10, y_split=10)

    for i in range(2, 101):
        save_picture_reduces_pixel_quality(pix, path_dir, used_steps=i)
