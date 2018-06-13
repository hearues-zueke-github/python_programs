#! /usr/bin/python2.7

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

img = Image.open("snow-mountains-1600x900-green-landscape-scenery-5k-5405.jpg")
img_resizes = img.resize((400, 225), Image.ANTIALIAS)
img_resizes.show()

# First get all parts!

pix = np.array(img)

# e.g tiles are 100x100

height, width, chans = pix.shape

tile_width = 100
pix_tiles = pix.transpose(0, 2, 1) \
               .reshape((height//tile_width, tile_width*chans, width)) \
               .transpose(0, 2, 1) \
               .reshape((height*width//tile_width//tile_width, tile_width, tile_width, chans)) \
               .transpose(0, 2, 1, 3) \
               .reshape((height*width//tile_width, tile_width, chans))
               # .reshape((height, width, chans)) \
               # .reshape((height*width//tile_width//tile_width, tile_width, tile_width, 3)) \

pix_1_tile = pix[:tile_width, :tile_width]
img_1_tile = Image.fromarray(pix_1_tile)
img_1_tile.save("tiles_1_tile.png", "PNG")

img_tiles = Image.fromarray(pix_tiles)
img_tiles.save("tiles_in_rows.png", "PNG")

pix_tiles = pix_tiles.reshape((height*width//tile_width//tile_width, tile_width, tile_width, chans))

bs = 20 # border_size
x, y, z, w = pix_tiles.shape
pix_tiles_borders = np.zeros((x, y+2*bs, z+2*bs, w), dtype=np.uint8)

pix_tiles_borders[:, bs:-bs, bs:-bs] = pix_tiles

pix_tiles_borders[:, :bs, bs:-bs] = pix_tiles[:, 0, :].reshape((x, 1, z, w))
pix_tiles_borders[:, -bs:, bs:-bs] = pix_tiles[:, -1, :].reshape((x, 1, z, w))
pix_tiles_borders[:, bs:-bs, :bs] = pix_tiles[:, :, 0].reshape((x, y, 1, w))
pix_tiles_borders[:, bs:-bs, -bs:] = pix_tiles[:, :, -1].reshape((x, y, 1, w))

pix_tiles_borders[:, :bs, :bs] = pix_tiles[:, 0, 0].reshape((x, 1, 1, 3))
pix_tiles_borders[:, :bs, -bs:] = pix_tiles[:, 0, -1].reshape((x, 1, 1, 3))
pix_tiles_borders[:, -bs:, :bs] = pix_tiles[:, -1, 0].reshape((x, 1, 1, 3))
pix_tiles_borders[:, -bs:, -bs:] = pix_tiles[:, -1, -1].reshape((x, 1, 1, 3))

img_pix_tiles_borders = Image.fromarray(pix_tiles_borders.reshape(height*width//tile_width//tile_width*(tile_width+bs*2), tile_width+2*bs, chans))
img_pix_tiles_borders.save("tiles_with_borders.png", "PNG")

pix_tiles_gray = pix_tiles_borders.astype(np.float). \
                                   dot([0.299, 0.587, 0.114])

pix_tiles_gray_int = pix_tiles_gray.copy().astype(np.uint8)
pix_tiles_gray_int[pix_tiles_gray >= 255.5] = 255

pix_tiles_integral = np.cumsum(np.cumsum(pix_tiles_gray_int.astype(np.float), axis=1), axis=2)
pix_tiles_integral = np.concatenate((np.zeros((x, 1, z+2*bs)).astype(np.float), pix_tiles_integral), axis=1)
pix_tiles_integral = np.concatenate((np.zeros((x, y+1+2*bs, 1)).astype(np.float), pix_tiles_integral), axis=2)

pix_tiles_integral -= np.min(pix_tiles_integral)
pix_tiles_integral /= np.max(pix_tiles_integral)
pix_tiles_integral *= 256
# pix_tiles_integral_int = pix_tiles_integral.copy().astype(np.uint8)
# pix_tiles_integral_int[pix_tiles_integral > 255.5] = 255
print("pix_tiles_integral[0, :10, :10]:\n{}".format(pix_tiles_integral[0, :10, :10]))
# img_tiles_integral = Image.fromarray(pix_tiles_integral_int)
img_tiles_integral = Image.fromarray(pix_tiles_integral.reshape((x*(y+1+bs*2), z+1+bs*2)))
if img_tiles_integral.mode != 'RGB':
    img_tiles_integral = img_tiles_integral.convert('RGB')
img_tiles_integral.save("tiles_integrals.png", "PNG")
