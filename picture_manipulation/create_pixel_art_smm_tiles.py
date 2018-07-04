#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os

import numpy as np

from PIL import Image

"""
    @param pix: is a numpy array with shape (x, y, z) (x%tw == 0, y%tw == 0)
    @param tw: tile width (also the tile height)

    @return: will return a numpy array with each square tiles separate
"""
def get_square_tiles(pix, tw):
    x, y, z = pix.shape
    return pix.transpose(0, 2, 1) \
              .reshape((x//tw, tw*z, y)) \
              .transpose(0, 2, 1) \
              .reshape((x*y//tw//tw, tw, tw, z)) \
              .transpose(0, 2, 1, 3) \
              .reshape((x//tw, y//tw, tw, tw, z)) # this is crazy!

def transform_square_tiles_to_img(pix, tw):
    a, b, c, d, e = pix.shape
    return pix.reshape((a*b, c, d, e)) \
              .transpose(0, 2, 1, 3) \
              .reshape((a, c*d, b*e)) \
              .transpose(0, 2, 1) \
              .reshape((a*c, e, b*d)) \
              .transpose(0, 2, 1) \

def get_approx_tiles(pix_img_tiles, tiles_row):
    shape = pix_img_tiles.shape
    print("pix_img_tiles.shape: {}".format(pix_img_tiles.shape))
    pix_img_tiles = pix_img_tiles.reshape((shape[:2]+(1, )+shape[2:])).astype(np.int64)
    print("pix_img_tiles.shape: {}".format(pix_img_tiles.shape))
    idx_tbl = np.argmin(np.sum(np.sum(np.sum((pix_img_tiles-tiles_row)**2, axis=-1), axis=-1), axis=-1), axis=-1)
    
    new_pix = tiles_row[idx_tbl]
    print("new_pix.shape: {}".format(new_pix.shape))

    return new_pix

if __name__ == "__main__":
    from os.path import expanduser
    home = expanduser("~")
    print("home: {}".format(home))

    path_images = home+"/Pictures/tiles_images/"

    img = Image.open(path_images+"smm_smb3_tiles.png")
    pix = np.array(img)

    tw = 16 # tile width
    th = 16 # tile height
    tiles_x = 10
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
                    [5, 38, 2, 3],

                    [0, 6, 0, 4],
                    [6, 6, 0, 5],
                    [7, 8, 0, 6],
                    [4, 2, 0, 7],
                    [0, 7, 0, 8],

                    [0, 6+16, 1, 4],
                    [6, 6+16, 1, 5],
                    [7, 8+16, 1, 6],
                    [4, 2+16, 1, 7],
                    [0, 7+16, 1, 8],

                    [0, 6+32, 2, 4],
                    [6, 6+32, 2, 5],
                    [7, 8+32, 2, 6],
                    [4, 2+32, 2, 7],
                    [0, 7+32, 2, 8]]
    
    # Fill the tiles into the pix_tiles
    for ts_y, ts_x, t_y, t_x in choose_tiles:
        pix_tiles[th*t_y:th*(t_y+1), tw*t_x:tw*(t_x+1)] = \
            pix[(th+1)*ts_y+1:(th+1)*ts_y+th+1, (tw+1)*ts_x+1:(tw+1)*ts_x+tw+1]

    img_tiles = Image.fromarray(pix_tiles)
    img_anime = Image.open(path_images+"nature_1.jpeg")
    # img_anime = Image.open(path_images+"anime_2.jpeg")
    # img_anime = Image.open(path_images+"anime_1.jpeg")
    
    pix_anime = np.array(img_anime)
    print("pix_anime.shape: {}".format(pix_anime.shape))
    anim_h, anim_w, anim_c = pix_anime.shape

    pix_anime = pix_anime[:(anim_h//th)*th, :(anim_w//tw)*tw]
    print("pix_anime.shape: {}".format(pix_anime.shape))

    # transform pix_tiles so that it will be like a
    # (tiles_y, tiles_x, th, tw, 3) shape
    x, y, z = pix_tiles.shape
    print("pix_tiles.shape: {}".format(pix_tiles.shape))

    # TODO: Fix this! (maybe)
    pix_tiles_own = get_square_tiles(pix_tiles, tw)
    pix_anime_own = get_square_tiles(pix_anime, tw)

    print("pix_tiles_own.shape: {}".format(pix_tiles_own.shape))
    print("pix_anime_own.shape: {}".format(pix_anime_own.shape))

    # pix_tiles_own = pix_tiles.transpose(0, 2, 1)
    # pix_tiles_own = pix_tiles_own.reshape((x//th, z*th, y)) # watch out the fail next time!
    # pix_tiles_own = pix_tiles_own.transpose(0, 2, 1)
    # pix_tiles_own = pix_tiles_own.reshape((x*y//th//tw, th, tw, z))
    # pix_tiles_own = pix_tiles_own.transpose(0, 2, 1, 3)
    # pix_tiles_own = pix_tiles_own.reshape((x//th, y//tw, th, tw, z))
    
    # pix_tiles_own = pix_tiles_own.reshape((x*y//tw, tw, z))

    # print("pix_tiles_own.shape: {}".format(pix_tiles_own.shape))
    # img_tiles_own = Image.fromarray(pix_tiles_own)
    # img_tiles_own.show()

    first_row = pix_tiles_own[0]
    # print("first_row.shape: {}".format(first_row.shape))

    # # TODO: add new snippet later
    # # print("first_row_col: "+f"{first_row_col}")

    # first_row_col = first_row.reshape((tiles_x*th, tw, 3))
    # img_row_col = Image.fromarray(first_row_col)

    # print("first_row_col.shape: {}".format(first_row_col.shape))
    # img_row_col.show()


    img_tiles.show()
    # img_anime.show()

    # pix_approx = pix_anime.copy()
    # pix_approx_tiles = get_square_tiles(pix_approx, tw)

    # Use this as a link (another reference!)
    # pix_approx_tiles[:] = get_approx_tiles(pix_anime_own, first_row).astype(np.uint8)
    pix_approx_tiles = get_approx_tiles(pix_anime_own, first_row).astype(np.uint8)
    # pix_approx = transform_square_tiles_to_img(pix_approx_tiles, tw)
    pix_approx = np.zeros((pix_approx_tiles.shape[0]*tw, pix_approx_tiles.shape[1]*tw, 3), dtype=np.uint8)
    pic_tiles_y = pix_approx_tiles.shape[0]
    pic_tiles_x = pix_approx_tiles.shape[1]
    for y in range(0, pic_tiles_y):
        for x in range(0, pic_tiles_x):
            pix_approx[tw*y:tw*(y+1), tw*x:tw*(x+1)] = pix_approx_tiles[y, x]

    print("pix_approx.shape: {}".format(pix_approx.shape))
    print("pix_approx.dtype: {}".format(pix_approx.dtype))
    # TODO: need to fix this!!!
    # img_approx = Image.fromarray(pix_approx_tiles[:2].reshape((2*16, 16*14, 3)))
    img_approx = Image.fromarray(pix_approx)

    # img_approx.show()
