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

def int_to_l_num(n, base=10):
    l = []
    while n > 0:
        l.append(n%base)
        n //= base
    return l


def digsum(n, base=10):
    return sum(int_to_l_num(n, base=base))


if __name__ == '__main__':
    base_max = 10
    n_max = 10
    power_max = 1000

    d_base_num_power_digsum = {}
    for base in range(2, base_max+1):
        print("base: {}".format(base))
        d_num_power_digsum = {}
        d_base_num_power_digsum[base] = d_num_power_digsum
        for n in range(1, n_max+1):
            d_power_digsum = {'power': [], 'digsum': []}
            d_num_power_digsum[n] = d_power_digsum

            if n%100==0:
                print("- n: {}".format(n))
            if n%10==0:
                continue

            p = 1
            for power in range(1, power_max+1):
                p *= n
                d_power_digsum['power'].append(power)
                d_power_digsum['digsum'].append(digsum(p, base=base))
                # d_power_digsum['digsum'].append(digsum(n**power, base=base))
