#! /usr/bin/python3.6

import dill
import os
import pdb
import sys

import numpy as np

from dotmap import DotMap
from PIL import Image

def resize_image(img, factor=2):
    h, w = img.size
    return img.resize((h*factor, w*factor))

if __name__ == "__main__":
    image_type = 2

    if image_type == 1: # random
        height = 128
        width = height
        pix = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        path_images = "images/random_sort_images_h_{}_w_{}/".format(height, width)
    elif image_type == 2:
        path_pic = "images/pexels-photo-236047.jpeg"
        # path_pic = "images/fall-autumn-red-season_resized.jpg"
        folder_prefix = path_pic.split("/")[-1].split(".")[0]
        pix = np.array(Image.open(path_pic))
        height, width = pix.shape[:2]
        path_images = "images/{}_sort_images_h_{}_w_{}/".format(folder_prefix, height, width)

    if not os.path.exists(path_images):
        os.makedirs(path_images)

    orders = [["f0", "f1", "f2"],
              ["f0", "f2", "f1"],
              ["f1", "f0", "f2"],
              ["f1", "f2", "f0"],
              ["f2", "f0", "f1"],
              ["f2", "f1", "f0"]]

    resize_image(Image.fromarray(pix)).save(path_images+"random_image_h_{}_w_{}.png".format(height, width))

    # TODO: sort first image with one type, and map to second sorted image!
    pixs_1d = []
    pix_v = pix.view("u1,u1,u1").reshape((-1, ))
    for i, order in enumerate(orders):
        pix2 = np.hstack((pix.reshape((-1, 3)), np.zeros((height*width, 8), dtype=np.uint8)))
        pix2 = pix2.view("u1,u1,u1,u8").reshape((-1, ))
        pix2["f3"] = np.arange(0, height*width)
        pix2 = np.sort(pix2, order=order+["f3"])

        # pix_sorted = np.sort(pix_v, order=order).view("u1")
        pix3 = pix2[["f0","f1","f2"]].copy().view("u1").reshape((height, width, 3))
        resize_image(Image.fromarray(pix3)).save(path_images+"random_image_sort_nr_{}_{}_h_{}_w_{}.png".format(i, "_".join(order), height, width))

        pixs_1d.append(pix2)

    for j, pix1 in enumerate(pixs_1d):
        path_images_2 = path_images+"/image_sorted_nr_{}/".format(j)
        if not os.path.exists(path_images_2):
            os.makedirs(path_images_2)
        print("j: {}".format(j))
        for i, pix2 in enumerate(pixs_1d):
            print("    i: {}".format(i))
            pix_t = np.zeros((height*width, 3), dtype=np.uint8)
            pix_t[pix1["f3"]] = pix2[["f0", "f1", "f2"]].copy().view("u1").reshape((height*width, 3))
            
            resize_image(Image.fromarray(pix_t.reshape((height, width, 3)))).save(path_images_2+"img_merge_j_{}_i_{}_h_{}_w_{}.png".format(j, i, height, width))
