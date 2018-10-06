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
    image_type = 1
    path_images = "images/comification/"
    if not os.path.exists(path_images):
        os.makedirs(path_images)
    
    if image_type == 1: # random
        height = 256
        width = height
        pix = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        file_name = "random_pic_h_{}_w_{}".format(height, width)
    elif image_type == 2:
        path_pic = "images/pexels-photo-236047.jpeg"
        # path_pic = "images/fall-autumn-red-season_resized.jpg"
        file_name = path_pic.split("/")[-1].split(".")[0]
        folder_prefix = path_pic.split("/")[-1].split(".")[0]
        pix = np.array(Image.open(path_pic))
        height, width = pix.shape[:2]

    file_path = path_images+file_name+".png"
    Image.fromarray(pix).save(file_path)

    pix_comb = np.hstack((pix.reshape((-1, 3)), np.zeros((height*width, 8), dtype=np.uint8)))
    pix_comb = pix_comb.view("u1,u1,u1,u8").reshape((-1, ))
    pix_comb["f3"] = np.sum(pix_comb[["f0", "f1", "f2"]].copy().view("u1").reshape((-1, 3)).astype(np.int)**2, axis=1)
    pix_comb = np.sort(pix_comb, order=["f3"]) # , "f0", "f1", "f2"])

    pix = pix_comb[["f0", "f1", "f2"]].copy().view("u1").reshape((height, width, 3))

    pix_unique = (lambda x: x[np.hstack(((True, ), x[:-1]!=x[1:]))])(pix_comb[["f0", "f1", "f2"]])

    n = pix_unique.shape[0]
    n_split = 10
    jump = int(n/(n_split-1))+1
    idxs = np.hstack((np.arange(0, n_split-1)*jump, (n-1, )))
    choosen_colors = pix_unique[idxs].copy().view("u1").reshape((-1, 3))
    choosen_colors_int = choosen_colors.astype(np.uint)

    print("Creating the comicfication of the image!")
    pix_comic = np.zeros(pix.shape, dtype=np.uint8)
    for y in range(0, height):
        print("y: {}".format(y))
        for x in range(0, width):
            rgb = pix[y, x]
            pix_comic[y, x] = choosen_colors[np.argmin(np.sum((choosen_colors_int-rgb)**2, axis=1))].astype(np.uint8)

    print("Finished comicfication!")
    Image.fromarray(pix_comic).save(path_images+file_name+"_comic.png")
