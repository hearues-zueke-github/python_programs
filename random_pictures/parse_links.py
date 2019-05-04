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

if __name__ == "__main__":
    # n = 4
    # for page_num in range(100*n, 100*(n+1)):
    # 1237
    # n = 1000
    n1 = 3800
    n2 = 4200
    for page_num in range(n1, n2):
        print("page_num: {}".format(page_num))
        os.system("curl https://pixabay.com/images/search/?pagi={} > page.html".format(page_num))
        
        f = open("page.html", "r")
        lines = f.readlines()
        f.close()

        l = "".join(lines) 
        finds = re.findall('gpj.*?//:sptth', l[::-1])

        img_links = list(map(lambda x: x[::-1], finds))
        img_links = sorted(list(set(img_links)))
        img_links = list(filter(lambda x: "340.jpg" in x, img_links))

        print("len(img_links): {}".format(len(img_links)))
        # print("img_links: {}".format(img_links))

        images_path = ROOT_PATH+"images/pixabay_com_9/"
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        lines = "\n".join(img_links)
        f = open("links.txt", "w")
        f.write(lines)
        f.close()

        current_path = os.getcwd()
        os.chdir(images_path)

        os.system("wget -q "+" ".join(img_links))

        # for file_link in img_links:
        #     print("Downloading file_link: {}".format(file_link))
        #     os.system("wget {link}".format(link=file_link))
        #     break

        # remove not needed files!
        root_dir_path, dir_names, file_names = next(os.walk(images_path))
        file_names_filtered = list(filter(lambda x: x.count(".") >= 2, file_names))
        print("len(file_names_filtered):\n{}".format(len(file_names_filtered)))
        if len(file_names_filtered) > 0:
            os.system("rm "+" ".join(file_names_filtered))
        
        root_dir_path, dir_names, file_names = next(os.walk(images_path))
        file_names_not_340_jpg = list(filter(lambda x: not "340.jpg" in x, file_names))
        print("len(file_names_not_340_jpg):\n{}".format(len(file_names_not_340_jpg)))
        if len(file_names_not_340_jpg):
            os.system("rm "+" ".join(file_names_not_340_jpg))

        os.chdir(current_path)

        # break
