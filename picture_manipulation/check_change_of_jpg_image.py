#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time
from functools import reduce

from collections import defaultdict

from PIL import Image

import numpy as np

sys.path.append("..")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style


class ShowImg(Frame, object):
    def __init__(self, img):
        parent = Tk()
        Frame.__init__(self, parent)
        self.pack(fill=BOTH, expand=1)
        label1 = Label(self)
        label1.photo= ImageTk.PhotoImage(img)
        label1.config(image=label1.photo)
        label1.pack(fill=BOTH, expand=1)
        parent.mainloop()


if __name__ == "__main__":
    dir_path = PATH_ROOT_DIR+"images/save_load_jpg_images/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path_template = dir_path+"image_{:02}.jpg"


    img1 = Image.open('images/autumn-colorful-colourful-33109.jpg')
    img1.save(file_path_template.format(0), quality=98, subsampling=2)

    pixs_parts = [np.array(img1)]
    # pixs_parts = [np.array(img1)[:30, :30]]

    bit_amount_per_byte = np.array(list(map(lambda x: bin(x).count("1"), range(0, 256))))
    # sys.exit(0)

    for i in range(0, 60):
        print("i: {}".format(i))
        img = Image.open(file_path_template.format(i))
        pix = np.array(img)
        pixs_parts.append(pix.copy())
        amount_different_bits = np.sum(bit_amount_per_byte[pixs_parts[-1]^pixs_parts[-2]])
        print("amount_different_bits: {}".format(amount_different_bits))
        # pixs_parts.append(pix[:30, :30].copy())
        v = pix[0, 0, 0]
        # pix[0, 0, 0] = (v&0xFE)|((v&0x1)^1)
        img2 = Image.fromarray(pix)
        img2.save(file_path_template.format(i+1), quality=100, subsampling=0)
