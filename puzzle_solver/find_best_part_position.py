#! /usr/bin/python2.7

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

img = Image.open("snow-mountains-1600x900-green-landscape-scenery-5k-5405.jpg")
img = img.resize((400, 225), Image.ANTIALIAS)
img.show()

pix = np.array(img)
pix_float = pix.astype(np.double)

height, width, _ = pix.shape

tile_size = 100
y_pos = np.random.randint(0, height-tile_size)
x_pos = np.random.randint(0, width-tile_size)
tile_patch = pix[y_pos:y_pos+tile_size,
                 x_pos:x_pos+tile_size]

print("y_pos: {}".format(y_pos))
print("x_pos: {}".format(x_pos))

img_patch = Image.fromarray(tile_patch)
img_patch.show()

tile_patch_sp = tile_patch.copy()
tile_patch_sp[np.random.randint(0, tile_size, (2, int(tile_size*0.4))).tolist()] = 0
tile_patch_sp[np.random.randint(0, tile_size, (2, int(tile_size*0.4))).tolist()] = 255

img_patch_sp = Image.fromarray(tile_patch_sp)
img_patch_sp.show()

score_board = np.zeros((height-tile_size, width-tile_size))
for y in xrange(0, height-tile_size):
    for x in xrange(0, width-tile_size):
        score_board[y, x] = np.sum((pix_float[y:y+tile_size, x:x+tile_size]-tile_patch)**2)/tile_size**2

y_best, x_best = np.unravel_index(score_board.argmin(), score_board.shape)
print("y_best: {}, x_best: {}".format(y_best, x_best))

score_board_new = score_board/np.max(score_board)
plt.figure()
plt.title("NO sp")
plt.imshow(score_board_new, interpolation="nearest")
plt.show()

score_board_sp = np.zeros((height-tile_size, width-tile_size))
for y in xrange(0, height-tile_size):
    for x in xrange(0, width-tile_size):
        score_board_sp[y, x] = np.sum((pix_float[y:y+tile_size, x:x+tile_size]-tile_patch_sp)**2)/tile_size**2

y_best, x_best = np.unravel_index(score_board_sp.argmin(), score_board_sp.shape)
print("y_best: {}, x_best: {}".format(y_best, x_best))

score_board_sp_new = score_board_sp/np.max(score_board_sp)
plt.figure()
plt.title("With sp")
plt.imshow(score_board_sp_new, interpolation="nearest")
plt.show()
