#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os

import numpy as np

from PIL import Image

import Utils


"""
    @param pix: numpay array of a image
    @param pxs: pixel size of one tile
    @return: returns a numpay array with the pixelated tiles
"""
def get_pixelated_pix(pix, pxs):
    height, width, _ = pix.shape
    pix_crop = pix[:(height//pxs)*pxs, :(width//pxs)*pxs]
    pix_crop_tiles = Utils.get_square_tiles(pix_crop, pxs)
    pix_crop_tiles_rgb = np.mean(np.mean(pix_crop_tiles, axis=-2), axis=-2).astype(np.uint8)

    pix_crop_pixel = np.zeros(pix_crop.shape, dtype=pix_crop.dtype)
    for y in range(0, pix_crop_tiles.shape[0]):
        for x in range(0, pix_crop_tiles.shape[1]):
            pix_crop_pixel[pixel_size*y:pixel_size*(y+1), pixel_size*x:pixel_size*(x+1)] = pix_crop_tiles_rgb[y, x]

    return pix_crop_pixel

if __name__ == "__main__":
    pixel_size = 16
    
    from os.path import expanduser
    home = expanduser("~")
    print("home: {}".format(home))

    path_images = home+"/Pictures/tiles_images/"

    file_name = "nature_1.jpeg"
    img_orig = Image.open(path_images+file_name)
    pix = np.array(img_orig)

    pixel_sizes = 2**np.arange(1, 6)
    for pixel_size in pixel_sizes:
        pix_crop_pixel = get_pixelated_pix(pix, pixel_size)
        img_crop = Image.fromarray(pix_crop_pixel)

        img_crop.save(path_images+file_name.replace(".jpeg", "pxs_{}.png".format(pixel_size)))

    # img_orig.show()
    # img_crop.show()


