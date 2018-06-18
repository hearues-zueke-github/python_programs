#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os

import numpy as np

from PIL import Image
    
if __name__ == "__main__":
    from os.path import expanduser
    home = expanduser("~")
    print("home: {}".format(home))

    path_images = home+"/Pictures/tiles_images/"

    img = Image.open(path_images+"smm_smb3_tiles.png")
    pix = np.array(img)

    tw = 16 # tile width
    th = 16 # tile height
    tiles_x = 4
    tiles_y = 3
    pix_tiles = np.zeros((th*tiles_y, tw*tiles_x, 3), dtype=np.uint8)

    # get all tiles per indices
    choose_tiles = [[0, 1, 0, 0],
                    [0, 2, 0, 1],
                    [0, 4, 0, 2],
                    [5, 6, 0, 3],
                    [0, 17, 1, 0],
                    [0, 18, 1, 1],
                    [0, 20, 1, 2],
                    [5, 22, 1, 3],
                    [0, 33, 2, 0],
                    [0, 34, 2, 1],
                    [0, 36, 2, 2],
                    [5, 38, 2, 3]]

    for ts_y, ts_x, t_y, t_x in choose_tiles:
        pix_tiles[th*t_y:th*(t_y+1), tw*t_x:tw*(t_x+1)] = \
            pix[(th+1)*ts_y+1:(th+1)*ts_y+th+1, (tw+1)*ts_x+1:(tw+1)*ts_x+tw+1]

    img_tiles = Image.fromarray(pix_tiles)
    img_tiles.show()

    img_anime = Image.open(path_images+"anime_1.jpeg")
    img_anime.show()

    pix_anime = np.array(img_anime)
    print("pix_anime.shape: {}".format(pix_anime.shape))
    anim_h, anim_w, anim_c = pix_anime.shape

    pix_anime = pix_anime[:(anim_h//th)*th, :(anim_w//tw)*tw]
    print("pix_anime.shape: {}".format(pix_anime.shape))

    # transform pix_tiles so that it will be like a
    # (tiles_y, tiles_x, th, tw, 3) shape
    x, y, z = pix_tiles.shape
    print("pix_tiles.shape: {}".format(pix_tiles.shape))

    pix_tiles_own = pix_tiles.transpose(0, 2, 1)
    pix_tiles_own = pix_tiles_own.reshape((th, z*x//th, y))
    pix_tiles_own = pix_tiles_own.transpose(0, 2, 1)
    pix_tiles_own = pix_tiles_own.reshape((x*y//th//tw, th, tw, z))
    pix_tiles_own = pix_tiles_own.transpose(0, 2, 1, 3)
    pix_tiles_own = pix_tiles_own.reshape((x//th, y//tw, th, tw, z))

    print("pix_tiles_own.shape: {}".format(pix_tiles_own.shape))
