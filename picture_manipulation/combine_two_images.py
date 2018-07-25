#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

from multiprocessing import Process, Queue
from PIL import Image

if __name__ == "__main__":
    from os.path import expanduser
    home = expanduser("~")
    print("home: {}".format(home))

    path_images = home+"/Pictures/combine_images/"
    # path_images = "/ramtemp/"
    path_images_output = "/ramtemp/"

    img_1 = Image.open(path_images+"nature_1.jpg")
    img_2 = Image.open(path_images+"nature_2.jpg")
    # img_1.show()
    # img_2.show()

    pix_1 = np.array(img_1)
    pix_2 = np.array(img_2)

    # mask_all_1 = 0x100-1
    # for i in range(0, 9):
    #     mask_1 = 2**i-1
    #     mask_2 = mask_all_1 ^ mask_1
    #     pix_comb = (pix_1&(mask_1)) | (pix_2&(mask_2))
    #     print("combine: mask_1: {:02X}, mask_2: {:02X}".format(mask_1, mask_2))
    #     img_comb = Image.fromarray(pix_comb)
    #     img_comb.save(path_images_output+"combined_nature_1_2_images_i_{}.png".format(i), "PNG")

    n = 40
    alphas = np.arange(0, n+1)/n

    pix_1_f = pix_1.astype(np.float)
    pix_2_f = pix_2.astype(np.float)

    for i, alpha in enumerate(alphas):
        pix_comb = (pix_1_f*alpha+pix_2_f*(1-alpha)).astype(np.uint8)
        print("combine: alpha: {:.2f}".format(alpha))
        img_comb = Image.fromarray(pix_comb)
        img_comb.save(path_images_output+"combined_nature_1_2_images_i_{}.png".format(i), "PNG")

