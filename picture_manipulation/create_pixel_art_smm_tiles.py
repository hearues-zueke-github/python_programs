#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os

import numpy as np

from PIL import Image

import Utils

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

def get_approx_tiles(pix_img_tiles, tiles_row, tiles_row_orig):
    shape = pix_img_tiles.shape
    print("pix_img_tiles.shape: {}".format(pix_img_tiles.shape))
    pix_img_tiles = pix_img_tiles.reshape((shape[:2]+(1, )+shape[2:])).astype(np.int64)
    print("pix_img_tiles.shape: {}".format(pix_img_tiles.shape))
    # idx_tbl = np.argmin(np.sum(np.sum(np.sum((pix_img_tiles-tiles_row)**2, axis=-1), axis=-1), axis=-1), axis=-1)
    idx_tbl = np.argmin(np.sum((pix_img_tiles-tiles_row)**2, axis=-1), axis=-1)
    
    # new_pix = tiles_row[idx_tbl]
    new_pix = tiles_row_orig[idx_tbl]
    print("new_pix.shape: {}".format(new_pix.shape))

    return new_pix

if __name__ == "__main__":
    from os.path import expanduser
    home = expanduser("~")
    print("home: {}".format(home))

    path_images = home+"/Pictures/tiles_images/"

    img = Image.open(path_images+"smm_smb1_tiles.png")
    pix = np.array(img)

    tw = 16 # tile width
    th = 16 # tile height
    tiles_x = 37+9
    tiles_y = 3
    pix_tiles = np.zeros((th*tiles_y, tw*tiles_x, 3), dtype=np.uint8)

    # get all tiles per indices
    choose_tiles = [[0, 1, 0, 0],
                    [0, 2, 0, 1],
                    [0, 4, 0, 2],
                    [5, 6, 0, 3],
                    [0, 6, 0, 4],
                    [6, 6, 0, 5],
                    [7, 8, 0, 6],
                    [4, 2, 0, 7],
                    [0, 7, 0, 8],
                    # red block
                    [3, 7, 0,  9],
                    [3, 8, 0, 10],
                    [3, 9, 0, 11],
                    [4, 7, 0, 12],
                    [4, 8, 0, 13],
                    [4, 9, 0, 14],
                    [6, 7, 0, 15],
                    [6, 8, 0, 16],
                    [6, 9, 0, 17],
                    # blue block
                    [3, 7+3, 0,  9+9],
                    [3, 8+3, 0, 10+9],
                    [3, 9+3, 0, 11+9],
                    [4, 7+3, 0, 12+9],
                    [4, 8+3, 0, 13+9],
                    [4, 9+3, 0, 14+9],
                    [6, 7+3, 0, 15+9],
                    [6, 8+3, 0, 16+9],
                    [6, 9+3, 0, 17+9],
                    # white block
                    [3, 7+6, 0,  9+18],
                    [3, 8+6, 0, 10+18],
                    [3, 9+6, 0, 11+18],
                    [4, 7+6, 0, 12+18],
                    [4, 8+6, 0, 13+18],
                    [4, 9+6, 0, 14+18],
                    [6, 7+6, 0, 15+18],
                    [6, 8+6, 0, 16+18],
                    [6, 9+6, 0, 17+18],

                    [2, 3, 0, 36],
                    [2, 4, 0, 37],
                    [2, 5, 0, 38],
                    [3, 3, 0, 36+3],
                    [3, 4, 0, 37+3],
                    [3, 5, 0, 38+3],
                    [4, 3, 0, 36+6],
                    [4, 4, 0, 37+6],
                    [4, 5, 0, 38+6],

                    [0, 17, 1, 0],
                    [0, 18, 1, 1],
                    [0, 20, 1, 2],
                    [5, 22, 1, 3],
                    [0, 6+16, 1, 4],
                    [6, 6+16, 1, 5],
                    [7, 8+16, 1, 6],
                    [4, 2+16, 1, 7],
                    [0, 7+16, 1, 8],
                    
                    [0, 33, 2, 0],
                    [0, 34, 2, 1],
                    [0, 36, 2, 2],
                    [5, 38, 2, 3],
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
    # img_anime = Image.open(path_images+"nature_1.jpeg")
    # img_anime = Image.open(path_images+"anime_2.jpeg")
    img_anime = Image.open(path_images+"anime_1.jpeg")

    baseheight = img_anime.size[0]*4
    hpercent = (baseheight/float(img_anime.size[1]))
    wsize = int((float(img_anime.size[0])*hpercent))
    img_anime = img_anime.resize((wsize, baseheight), Image.ANTIALIAS)

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

    # TODO: Fix this! (maybe)
    pix_tiles_own = get_square_tiles(pix_tiles, tw).astype(np.float)
    pix_anime_own = get_square_tiles(pix_anime, tw).astype(np.float)

    print("pix_tiles_own.shape: {}".format(pix_tiles_own.shape))
    print("pix_anime_own.shape: {}".format(pix_anime_own.shape))

    print("pix_tiles_own.dtype: {}".format(pix_tiles_own.dtype))
    print("pix_anime_own.dtype: {}".format(pix_anime_own.dtype))

    # pix_tiles_own_reshape = pix_tiles_own.reshape((lambda x: (x[0], x[1], x[2]*x[3], x[4]))(pix_tiles_own.shape))

    get_mean_pix = lambda pix: np.mean(np.mean(pix, axis=-2), axis=-2)
    get_norm_pix = lambda pix: (lambda pix, min_val, max_val: (pix-min_val)*255./(max_val-min_val))(
        *(pix, np.min(pix), np.max(pix)))


    pix_tiles_own_mean = get_mean_pix(pix_tiles_own)
    pix_anime_own_mean = get_mean_pix(pix_anime_own)

    pix_tiles_own_norm = get_norm_pix(pix_tiles_own_mean)
    pix_anime_own_norm = get_norm_pix(pix_anime_own_mean)

    print("np.min(pix_tiles_own, axis=-1): {}".format(np.min(pix_tiles_own, axis=-1)))

    first_row = pix_tiles_own[0]
    first_row_mean = get_mean_pix(first_row)
    first_row_norm = get_norm_pix(first_row_mean)

    img_tiles.show()

    pix_approx_tiles = get_approx_tiles(pix_anime_own_norm, first_row_norm, first_row).astype(np.uint8)
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

    img_approx.show()

    # pix_crop_tiles_norm, pix_tiles_line_norm = Utils.get_pixelated_tiled_pix(pix_anime, tw, first_row)
    pix_crop_filled = Utils.get_pixelated_tiled_pix(pix_anime, tw, first_row)
    img_crop_filled = Image.fromarray(pix_crop_filled)

    img_crop_filled.show()
