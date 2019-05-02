#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

from PIL import Image

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__ == "__main__":
    w1 = 40
    h1 = 30
    
    images_dir_path = ROOT_PATH+"images/pixabay_com/pngs/{w1}x{h1}/".format(w1=w1, h1=h1)

    root_path_dir, dir_names, file_names = next(os.walk(images_dir_path))

    m = 100
    n = 80

    assert m*n <= len(file_names), "Needed {} files, but only {} files where found!".format(m*n, len(file_names))

    w2 = w1*m
    h2 = h1*n

    output_dir_path = ROOT_PATH+"images/combined_small_images/"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    pix_all = np.empty((h2, w2, 3), dtype=np.uint8)

    using_idx = np.random.permutation(np.arange(0, len(file_names)))[:m*n]
    print("using_idx: {}".format(using_idx))


    print("Get all images into pixses array!")
    pixses = np.empty((m*n, h1, w1, 3), dtype=np.uint8)
    # for i in range(0, m*n):
    for i, pic_nr in enumerate(using_idx, 0):
        file_name = file_names[pic_nr]
        img = Image.open(root_path_dir+file_name)
        pix_part = np.array(img)
        try:
            if len(pix_part.shape) == 2:
                
                # pix_part = np.tile(pix_part, 3).T.reshape((h1, w2, 3))
                pix_part = np.tile(pix_part.reshape((-1, )), 3).reshape((3, -1)).T.reshape((h1, w1, 3))
            pixses[i] = pix_part
        except:
            print("A problem with the file '{}' in '{}'!".format(file_name, root_path_dir))
            sys.exit(-1)

    print("Combine all pixses to one image together!")
    for j in range(0, n):
        print("j: {}".format(j))
        for i in range(0, m):
            pix_all[h1*j:h1*(j+1), w1*i:w1*(i+1)] = pixses[j*m+i]

    save_extension = "jpg"

    img_all = Image.fromarray(pix_all)
    img_all.save(output_dir_path+"combines_images_w1_{w1}_h1_{h1}_m_{m}_n_{n}_not_sorted.{extension}".format(w1=w1, h1=h1, m=m, n=n, extension=save_extension))


    print("Sort images by pixel sum!")
    idx_sort = np.argsort(np.sum(pixses.reshape((m*n, -1)), axis=1))
    pixses = pixses[idx_sort]

    print("Combine all pixses sorted to one image together!")
    for j in range(0, n):
        print("j: {}".format(j))
        for i in range(0, m):
            pix_all[h1*j:h1*(j+1), w1*i:w1*(i+1)] = pixses[j*m+i]

    img_all = Image.fromarray(pix_all)
    img_all.save(output_dir_path+"combines_images_w1_{w1}_h1_{h1}_m_{m}_n_{n}_sorted_pixses.{extension}".format(w1=w1, h1=h1, m=m, n=n, extension=save_extension))
