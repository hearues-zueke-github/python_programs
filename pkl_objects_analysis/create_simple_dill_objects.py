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
# from multiprocessing import shared_memory # in python3.8 available!

import matplotlib
import matplotlib.pyplot as plt

import subprocess

import numpy as np

from PIL import Image

from copy import copy, deepcopy

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

def print_hex_from_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    data = data.hex().upper()
    if len(data) % 2:
        data = "0"+data
    l = [" "+data[2*i:2*(i+1)] for i in range(0, len(data)//2)]
    lines = [''.join(l[16*i:16*(i+1)]) for i in range(0, (len(l)+16-len(l)%16)//16-(len(l)%16==0 and len(l)>=16))]
    # print("data:\n{}".format(data))
    for i, line in enumerate(lines, 0):
        print("{:04X}:{}".format(i*16, line))


if __name__ == "__main__":
    print("Hello World!")

    # create a int object
    a = (0x100000000, 0x1000000)
    with open('a_1.pkl', 'wb') as f:
        dill.dump(a, f)

    print_hex_from_file('a_1.pkl')
