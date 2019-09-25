#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import mmap
import os
import re
import sys
import time

import itertools
import multiprocessing

from multiprocessing import Process, Pipe

import matplotlib
import matplotlib.pyplot as plt

import subprocess

import numpy as np

from PIL import Image

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__ == "__main__":
    argv = sys.argv

    if len(argv) < 4:
        print("Not enough arguments:")
        print("usage: python3 xor_images.py <path_picture_1> <path_picture_2> <output_picture>")
        sys.exit(-1)

    path_1 = argv [1]
    path_2 = argv [2]
    path_out = argv [3]

    if not os.path.exists(path_1):
        print("Path for image '{}' does not exists!".format(path_1))
    if not os.path.exists(path_2):
        print("Path for image '{}' does not exists!".format(path_2))

    img1 = Image.open(path_1)
    img2 = Image.open(path_2)

    pix1 = np.array(img1)
    pix2 = np.array(img2)

    pix_out = pix1 ^ pix2

    img_out = Image.fromarray(pix_out)
    img_out.save(path_out)

    print("Saved XOR image to '{}'!".format(path_out))
