#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import io
import datetime
import os
import pathlib
import re
import shutil
import string
import subprocess
import sys
import time
import mmap

import numpy as np
import multiprocessing as mp

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

if __name__ == '__main__':
    # n = 4
    l_amount = []
    l_amount_diags = []
    # for n in range(3, 4):
    for n in range(1, 31):
        print("n: {}".format(n))
        l = []
        l_diags = []
        # for w = 1
        for j in range(1, n+1):
            for i in range(1, n+1):
                l.append((i, j, (n-i+1)*(n-j+1)))

        for w in range(2, n+1):
            for a in range(1, w):
                b = w - a
                if (a > 1) and (b > 1) and (a%b==0 or b%a==0):
                    continue

                for j1 in range(1, ((n-b)//a)+1):
                    for i1 in range(1, ((n-a)//b)+1):
                        x = a*i1+b*j1
                        y = b*i1+a*j1
                        if x>n or y>n:
                            continue
                        l_diags.append(({'ab': (a, b), 'i1j1': (i1, j1)}, (x, y), (n-x+1)*(n-y+1)))


        # print("l: {}".format(l))
        print("- len(l): {}".format(len(l)))
        print("- len(l_diags): {}".format(len(l_diags)))

        amount = sum([i for _, _, i in l])
        print("- amount: {}".format(amount))
        amount_diags = sum([i for _, _, i in l_diags])
        print("- amount_diags: {}".format(amount_diags))

        l_amount.append(amount)
        l_amount_diags.append(amount_diags)

    print("l_amount: {}".format(l_amount))
    print("l_amount_diags: {}".format(l_amount_diags))

    l_amount_sum = [i+j for i, j in zip(l_amount, l_amount_diags)]

    # A113751
    # l_amount_diags

    # A085582
    # l_amount_sum
