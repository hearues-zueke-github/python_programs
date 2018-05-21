#! /usr/bin/python2.7

import numpy as np

import os

import sys
sys.path.append('..')
# sys.path.append('../encryption')

from encryption import Utils

arr_rnd = np.random.randint(0, 256, (1024*1024*1, )).astype(np.uint8)
# print("arr_rnd:")
# Utils.pretty_block_printer(arr_rnd, 8, len(arr_rnd))

# path = "/home/doublepmcl/Document"
# os.chdir(path)

with open("/home/doublepmcl/Documents/random_data.hex", "wb") as fout:
    arr_rnd.tofile(fout)
