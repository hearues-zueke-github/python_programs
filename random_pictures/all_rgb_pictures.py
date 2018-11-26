#! /usr/bin/python2.7

import os
import sys

import numpy as np

from copy import deepcopy
from dotmap import DotMap

from PIL import Image, ImageDraw, ImageFont

def get_all_rgb_pix():
    width = 4096
    height = 4096

    pix_part = np.zeros((256, 256, 3)).astype(np.uint8)
    pix_part[:, :, 0] = np.arange(0, 256).reshape((1, -1))
    pix_part[:, :, 1] = np.arange(0, 256).reshape((-1, 1))

    pix_all_rgb = np.zeros((height, width, 3)).astype(np.uint8)
    for y in xrange(0, 16):
        for x in xrange(0, 16):
            pix_all_rgb[256*y:256*(y+1), 256*x:256*(x+1)] = pix_part
            pix_all_rgb[256*y:256*(y+1), 256*x:256*(x+1), 2] = y*16+x

    return pix_all_rgb

def find_most_approx_all_rgb_picture(file_name):
    img = Image.open(file_name)
    pix_all_rgb = get_all_rgb_pix().astype(np.int)

    print("mix rows")
    for i in xrange(0, 4096):
        idx = np.random.permutation(np.arange(0, 4096))
        pix_all_rgb[i] = pix_all_rgb[i, idx]
    print("mix cols")
    for i in xrange(0, 4096):
        idx = np.random.permutation(np.arange(0, 4096))
        pix_all_rgb[:, i] = pix_all_rgb[idx, i]

    pix = np.array(img).astype(np.int)

    changed_pix = 0
    # scale = 4
    for k in xrange(0, 10):
        for _ in xrange(0, 100000):
            # x1, y1 = np.random.randint(0, 4096//scale, (2, ))
            # x2, y2 = np.random.randint(0, 4096//scale, (2, ))
            x1, y1 = np.random.randint(0, 4096, (2, ))
            x2, y2 = np.random.randint(0, 4096, (2, ))
            while x1 == x2 and y1 == y2:
                x2, y2 = np.random.randint(0, 4096//scale, (2, ))
                # x2, y2 = np.random.randint(0, 4096, (2, ))

            # rgb_1 = pix_all_rgb[scale*y1:scale*(y1+1), scale*x1:scale*(x1+1)].copy()
            # rgb_2 = pix_all_rgb[scale*y2:scale*(y2+1), scale*x2:scale*(x2+1)].copy()
            # rgb_orig_1 = pix[scale*y1:scale*(y1+1), scale*x1:scale*(x1+1)].copy()
            # rgb_orig_2 = pix[scale*y2:scale*(y2+1), scale*x2:scale*(x2+1)].copy()

            rgb_1 = pix_all_rgb[y1, x1]
            rgb_2 = pix_all_rgb[y2, x2]
            rgb_orig_1 = pix[y1, x1]
            rgb_orig_2 = pix[y2, x2]

            if np.sum((rgb_1-rgb_orig_2)**2) < np.sum((rgb_1-rgb_orig_1)**2) and \
               np.sum((rgb_2-rgb_orig_1)**2) < np.sum((rgb_2-rgb_orig_2)**2):
               # pix_all_rgb[scale*y1:scale*(y1+1), scale*x1:scale*(x1+1)] = rgb_2
               # pix_all_rgb[scale*y2:scale*(y2+1), scale*x2:scale*(x2+1)] = rgb_1
               pix_all_rgb[y1, x1] = rgb_2
               pix_all_rgb[y2, x2] = rgb_1
               changed_pix += 1

        print("k: {}, changed_pix: {}".format(k, changed_pix))
        img_new = Image.fromarray(pix_all_rgb.astype(np.uint8))
        img_new.save(file_name.replace(".jpg", "_iter_{}.png".format(k)), "PNG")

if __name__ == "__main__":
    pix_all_rgb = get_all_rgb_pix()

    path_images = "images/"
    if not os.path.exists(path_images):
        os.makedirs(path_images)

    img = Image.fromarray(pix_all_rgb)
    img.save(path_images+"all_rgb.png", "PNG")
    img_resize = img.resize((512, 512), Image.ANTIALIAS)
    img_resize.save(path_images+"all_rgb_resize.png", "PNG")

    # file_name = "SDvision_ramses_disksAMR8HR00019_redToWhite_4096x4096.jpg"
    # find_most_approx_all_rgb_picture(file_name)
