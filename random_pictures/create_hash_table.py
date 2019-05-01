#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import re
import sys
import time

import itertools
import multiprocessing

import matplotlib
import matplotlib.pyplot as plt

import filehash
import subprocess

cpu_amount = multiprocessing.cpu_count()

import numpy as np

from PIL import Image, ImageFile

hasher = filehash.FileHash('sha512')

# import decimal
# prec = 50
# decimal.getcontext().prec = prec

# from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

def get_image_size_from_path(file_path):
    ImPar=ImageFile.Parser()
    with open(file_path, "rb") as f:
        ImPar=ImageFile.Parser()
        chunk = f.read(2048)
        count=2048
        while chunk != "":
            ImPar.feed(chunk)
            if ImPar.image:
                break
            chunk = f.read(2048)
            count+=2048
        # print(ImPar.image.size)
        # print(count)
    return ImPar.image.size

"""
public internet pictures can be see at:
https://pixabay.com/images/search/
https://www.publicdomainpictures.net/en/latest-pictures.php?page=0
https://isorepublic.com/
etc.

general:
https://en.99designs.at/blog/resources/public-domain-image-resources/
"""

if __name__ == "__main__":
    images_path = ROOT_PATH+'images/pixabay_com/'

    images_path_pngs = images_path+'pngs/'
    if not os.path.exists(images_path_pngs):
        os.makedirs(images_path_pngs)

    root_path_dir, dir_names, file_names = next(os.walk(images_path))
    print("root_path_dir: {}".format(root_path_dir))
    print("len(dir_names): {}".format(len(dir_names)))
    print("len(file_names): {}".format(len(file_names)))

    lines = []
    for file_nr, file_name in enumerate(file_names, 0):
        if not "." in file_name:
            continue
        
        img_orig_file_path = root_path_dir+file_name
        h_512 = hasher.hash_file(img_orig_file_path).upper()
        lines.append("{h_512},{file_name}".format(h_512=h_512, file_name=file_name))

    f = open("hash_table_for_files_in_pixabay_com.txt", "w")
    f.write("\n".join(lines)+"\n")
    f.close()
