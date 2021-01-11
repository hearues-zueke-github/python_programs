#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

# import tempfile
from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

from collections import defaultdict
from copy import deepcopy
from dotmap import DotMap
from operator import itemgetter

from pprint import pprint

from os.path import expanduser

import itertools

import multiprocessing as mp

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
import global_object_getter_setter
import utils

from bit_automaton import BitAutomaton

def range_gen(g, n):
    i = 0
    while (v := next(g)) != None:
        yield v
        i += 1
        if i == n:
            break


if __name__ == '__main__':
    path_images = PATH_ROOT_DIR + 'images/'

    file_name = 'nature_trees_river_84761_800x600.jpg'
    image_path = path_images + file_name

    path_images_save = TEMP_DIR + 'save_images/{}/'.format(file_name.split('.')[0])
    utils.mkdirs(path_images_save)

    img = Image.open(image_path)
    pix = np.array(img)

    funcs_str = """
def rng(seed=0):
    a = seed
    while True:
        a = ((a + 19) ^ 0x2343) % 15232
        yield a % 13
def fun_0():
    v1 = (u + dl) % 2 == 0
    v2 = (u + dr) % 3 == 0
    v3 = (ur + l) % 2 == 0
    return v1 | v2 | v3
def fun_1():
    return (u + dl) % 3 == 0
def fun_2():
    return ((u + dl + urr) % 2 == 0) ^ ((dr + dll + uurr) % 2 == 0)
def fun_3():
    a = u+d+l+r+ur+ul+dr+dl
    return (a==2)|(a==3)
def fun_4():
    return n == 1
"""

    # h = 3
    # w = 4

    # frame = 2
    # frame_wrap = True

    # arr = np.random.randint(0, 0x100, (h, w), dtype=np.uint8)
    # arr_bits = np.array([[list(map(int, bin(v)[2:].zfill(8))) for v in row] for row in arr], dtype=np.uint8).transpose(2, 0, 1)

    # amount_bit_automaton = arr_bits.shape[0]

    # l_bit_automaton = [BitAutomaton(h=h, w=w, frame=frame, frame_wrap=frame_wrap, funcs_str=funcs_str) for _ in range(0, amount_bit_automaton)]

    # for bit_automaton, bits in zip(l_bit_automaton, arr_bits):
    #     bit_automaton.set_field(bits)

    # pix = np.random.randint(0, 0x100, (400, 500, 3), dtype=np.uint8)

    h, w = pix.shape[:2]

    frame = 2
    frame_wrap = True

    # arr = np.random.randint(0, 0x100, (h, w), dtype=np.uint8)
    arr_bits = np.array([[list(itertools.chain(*[list(map(int, bin(b)[2:].zfill(8))) for b in v])) for v in row] for row in pix], dtype=np.uint8).transpose(2, 0, 1)
    # sys.exit()

    amount_bit_automaton = arr_bits.shape[0]

    l_bit_automaton = [BitAutomaton(h=h, w=w, frame=frame, frame_wrap=frame_wrap, funcs_str=funcs_str) for _ in range(0, amount_bit_automaton)]

    for bit_automaton, bits in zip(l_bit_automaton, arr_bits):
        bit_automaton.set_field(bits)

    def convert_bit_field_to_pix(l_bit_automaton):
        assert len(l_bit_automaton) == 24 # 24 bits

        arr = np.array([np.sum([bit_automaton<<j for j, bit_automaton in zip(range(7, -1, -1), l_bit_automaton[8*i:8*(i+1)])], axis=0) for i in range(0, 3)], dtype=np.uint8)
        return arr.transpose(1, 2, 0)

    pix2 = convert_bit_field_to_pix(l_bit_automaton)
    assert np.all(pix2 == pix)

    print("i: {}".format(0))
    Image.fromarray(pix2).save(path_images_save + '{:04}.png'.format(0))

    for i in range(1, 100):
        print("i: {}".format(i))
        for bit_automaton in l_bit_automaton:
            bit_automaton.execute_func(3)
        pix2 = convert_bit_field_to_pix(l_bit_automaton)
        Image.fromarray(pix2).save(path_images_save + '{:04}.png'.format(i))
