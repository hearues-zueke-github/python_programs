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

def prepare_functions(func_str):
    bit_automaton = BitAutomaton(h=10, w=8, frame=2, frame_wrap=True, l_func=[], func_rng=None)
    
    d_vars = bit_automaton.d_vars
    field_size = bit_automaton.field_size
    d_func = {}
    exec(funcs_str, d_vars, d_func)

    assert 'rng' in d_func.keys()

    # test if each function starting with 'fun_' is returning a boolean array!
    l_func_name = []
    for func_name in d_func.keys():
        if func_name[:4] != 'fun_':
            continue

        l_func_name.append(func_name)

        # print("func_name: {}".format(func_name))
        v = d_func[func_name]()
        assert v.dtype == np.bool
        assert v.shape == field_size

    # check, if every function name is appearing starting from 0 to upwards in ascending order!
    assert np.all(np.diff(np.sort([int(v.replace('fun_', '')) for v in l_func_name])) == 1)

    l_func = [d_func[func_name] for func_name in l_func_name]
    # s_func_nr = set(range(0, len(l_func_name)))
    func_rng = d_func['rng']

    return l_func, func_rng


if __name__ == '__main__':
    path_images = PATH_ROOT_DIR + 'images/'

    file_name = 'nature_trees_river_84761_800x600.jpg'
    image_path = path_images + file_name

    path_images_save = TEMP_DIR + 'save_images/{}/'.format(file_name.split('.')[0])
    utils.mkdirs(path_images_save)

    img = Image.open(image_path)
    pix = np.array(img)

    funcs_str_rng = """
def rng(seed=0):
    a = seed
    while True:
        a = ((a + 19) ^ 0x2343) % 15232
        yield a % 13
"""

    funcs_str_func = """
def inv(x):
    return ~x
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
def fun_5():
    return l == 1
def fun_6():
    a1 = r & u & ur
    a2 = n & ull & ul
    a3 = u & d & l & r
    a4 = ll & dl & r
    return (a1 | a2 | a3 | a4) == 1
def fun_7():
    a1 = r & u & ur
    a2 = n & ull & ul
    a3 = u & d & l & r
    a4 = ll & dl & r
    return (a1 + a2 + a3 + a4) >= 2
"""
    
    # funcs_str = funcs_str_rng + funcs_str_func

    frame = 2
    frame_wrap = True
    
    funcs_str_func = "def inv(x):\n return ~x\n"

    l_vars_x = [(i)]
    l_vars = []

    bit_automaton = BitAutomaton(h=10, w=8, frame=2, frame_wrap=True, l_func=[], func_rng=None)

    funcs_str = funcs_str_rng + funcs_str_func
    sys.exit()

    # TODO: create some random funcstion too!

    l_func, func_rng = prepare_functions(funcs_str)

    h, w = pix.shape[:2]


    arr_bits = np.array([[list(itertools.chain(*[list(map(int, bin(b)[2:].zfill(8))) for b in v])) for v in row] for row in pix], dtype=np.uint8).transpose(2, 0, 1)

    amount_bit_automaton = arr_bits.shape[0]

    l_bit_automaton = [BitAutomaton(h=h, w=w, frame=frame, frame_wrap=frame_wrap, l_func=l_func, func_rng=func_rng) for _ in range(0, amount_bit_automaton)]

    for bit_automaton, bits in zip(l_bit_automaton, arr_bits):
        bit_automaton.set_field(bits.astype(np.bool))

    def convert_bit_field_to_pix(l_bit_automaton):
        assert len(l_bit_automaton) == 24 # 24 bits

        arr = np.array([np.sum([bit_automaton<<j for j, bit_automaton in zip(range(7, -1, -1), l_bit_automaton[8*i:8*(i+1)])], axis=0) for i in range(0, 3)], dtype=np.uint8)
        return arr.transpose(1, 2, 0)

    pix2 = convert_bit_field_to_pix(l_bit_automaton)
    assert np.all(pix2 == pix)

    print("i: {}".format(0))
    Image.fromarray(pix2).save(path_images_save + '{:04}.png'.format(0))

    l_func_nr = [6, 7]
    for i in range(1, 25):
    # for i in range(1, 100):
        func_nr = l_func_nr[i % 2]
        print("i: {}, func_nr: {}".format(i, func_nr))

        for bit_automaton in l_bit_automaton:
            bit_automaton.execute_func(5)
            # bit_automaton.execute_func(func_nr)
        pix2 = convert_bit_field_to_pix(l_bit_automaton)
        Image.fromarray(pix2).save(path_images_save + '{:04}.png'.format(i))
