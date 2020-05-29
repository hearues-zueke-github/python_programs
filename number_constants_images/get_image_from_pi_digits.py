#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import pdb
import sys
import time

import dill

# Needed for DataFrame (databases etc.)
import numpy as np
import pandas as pd
import pandas.io.sql as sql

from functools import reduce

from dotmap import DotMap

from PIL import Image

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"
USER_HOME_PATH = os.path.expanduser('~')+'/'
# print("USER_HOME_PATH: {}".format(USER_HOME_PATH))


def convert_num_to_base_num(n, b):
    def gen(n):
        while n > 0:
            yield n%b; n //= b
    return [i for i in gen(n)][::-1]


def amount_digits_of_num(n, b):
    i = 0
    while n > 0:
        i+=1;n//=b
    return i


def get_num_from_base_lst(l, b):
    n = 0
    mult = 1
    for i, v in enumerate(reversed(l), 0):
        n += v*mult
        mult *= b
    return n


def convert_num_commas_to_base_num(n, b_src, length, b_dst, length2):
    n_comma = b_src**length
    assert n < n_comma
    digits = []
    i = 0
    while i < length2:
        n *= b_dst
        j = n//n_comma
        digits.append(j)
        n -= j*n_comma
        i += 1
    return digits


def convert_num_commas_to_base_num_faster(n, b_src, length, b_dst, length2):
    n_comma = b_src**length
    assert n < n_comma

    


def remove_zeros_from_right(l):
    for i, v in enumerate(reversed(l), 0):
        if v != 0:
            break
    return l[:-i]


assert convert_num_to_base_num(10, 10)==[1, 0]
assert convert_num_to_base_num(102, 10)==[1, 0, 2]
assert convert_num_to_base_num(0b10011, 2)==[1, 0, 0, 1, 1]
assert convert_num_to_base_num(0o170125, 8)==[1, 7, 0, 1, 2, 5]
assert convert_num_to_base_num(0x1F511, 16)==[1, 15, 5, 1, 1]

assert amount_digits_of_num(0b1001011, 2)==7
assert amount_digits_of_num(123, 10)==3
assert amount_digits_of_num(0o1234, 8)==4
assert amount_digits_of_num(0x12356, 16)==5

assert get_num_from_base_lst([1, 2, 3], 10)==123
assert get_num_from_base_lst([1, 2, 3, 6, 9, 0], 10)==123690
assert get_num_from_base_lst([1, 0, 1, 1, 0], 2)==0b10110
assert get_num_from_base_lst([1, 7, 7, 3], 8)==0o1773
assert get_num_from_base_lst([1, 14, 12, 0, 8], 16)==0x1EC08

assert convert_num_commas_to_base_num(10, 10, 2, 2, 15)==[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
assert convert_num_commas_to_base_num(25, 10, 2, 2, 5)==[0, 1, 0, 0, 0]
assert convert_num_commas_to_base_num(26, 10, 2, 8, 10)==[2, 0, 5, 0, 7, 5, 3, 4, 1, 2]
assert convert_num_commas_to_base_num(625, 10, 5, 16, 10)==[0, 1, 9, 9, 9, 9, 9, 9, 9, 9]

digits_base64 = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_")
dict_base64 = {k: i for i, k in enumerate(digits_base64, 0)}
int_lst_to_num64 = lambda x: "".join(list(map(lambda y: digits_base64[y], x)))
num64_to_int_lst = lambda x: list(map(lambda y: dict_base64[y], x))

if __name__ == "__main__":
    # sys.exit(0)

    # print("Hello World!")
    path_file = USER_HOME_PATH+"Documents/Pi - Hex - Chudnovsky.txt"

    with open(path_file, 'r') as f:
        line = f.readlines()[0]
        # line[2:]

    a = line[2:100]
    dec_commas = np.array(convert_num_commas_to_base_num(int(a, 16), 16, len(a), 10, 200))
    print("a: {}".format(a))
    print("dec_commas: {}".format(dec_commas))

    s = line[2:100] 
    num_64_lst = convert_num_commas_to_base_num(int(s, 16), 16, len(s), 64, 100)
    num_64_lst_strip = remove_zeros_from_right(num_64_lst)
    num_64 = int_lst_to_num64(num_64_lst_strip)
    print("num_64: {}".format(num_64))

    s2 = int_lst_to_num64(convert_num_commas_to_base_num(get_num_from_base_lst(num64_to_int_lst(num_64), 64), 64, len(num_64), 16, 98))
    print("s2: {}".format(s2))

    # l = 100000
    # num = int(line[2:l+2], 16)
    # nums_comma_base = {}
    # for b in range(2, 21):
    #     print("b: {}".format(b))
    #     # arr = convert_num_commas_to_base_num(num, 16, l, b, l)
    #     arr = np.array(convert_num_commas_to_base_num(num, 16, l, b, l), dtype=np.uint8)
    #     nums_comma_base["base_{}".format(b)] = arr

    # with open("pi_commas_bases.pkl", "wb") as f:
    #     dill.dump(nums_comma_base, f)

    with open("pi_commas_bases.pkl", "rb") as f:
        nums_comma_base = dill.load(f)
    
    # for key in nums_comma_base:
    #     nums_comma_base[key] = np.array(nums_comma_base[key], dtype=np.uint8)

    # with open("pi_commas_bases.pkl", "wb") as f:
    #     dill.dump(nums_comma_base, f)

    sys.exit(0)


    four_bit_values = np.array(list(map(lambda x: int(x, 16), line[2:])))
    byte_values = np.sum(four_bit_values.reshape((-1, 2))*16**np.array([1, 0]), axis=1)

    colors_int = np.array([
        0x000000,
        0x000080,
        0x0000FF,
        0x008000,
        0x00FF00,
        0x800000,
        0xFF0000,
        0x00FF80,
        0x00FFFF,
        0xFF0080,
        0xFF00FF,
        0xFF8000,
        0xFFFF00,
        0xFF8080,
        0xFFFF80,
        0xFFFFFF,
    ])

    colors_r = (colors_int>>16)&0xFF
    colors_g = (colors_int>>8)&0xFF
    colors_b = (colors_int>>0)&0xFF
    colors_rgb = np.vstack([colors_r, colors_g, colors_b]).T

    colors_rgb = np.zeros((2, 3), dtype=np.uint8)
    colors_rgb[1] = 255

    # colors_rgb = np.zeros((3, 3), dtype=np.uint8)
    # colors_rgb[1] = 128
    # colors_rgb[2] = 255
    
    # colors_rgb = (((np.zeros((16, 3))+np.arange(0, 16).reshape((-1, 1)))/16)*256).astype(np.uint8)

    resize_factor = 5

    dir_images = PATH_ROOT_DIR+"images/"
    if not os.path.exists(dir_images):
        os.makedirs(dir_images)

    num = int(line[2:10000], 16)
    base_num_lst = convert_num_to_base_num(num, 2)
    base_num_arr = np.array(base_num_lst)

    n = 100
    indices_lsts = list(zip(*[(np.arange(0, i).tolist(), np.arange(i-1, -1, -1).tolist()) for i in range(1, n+1)]))
    indices = tuple(map(lambda x: np.array(reduce(lambda a, b: a+b, x, [])), indices_lsts))

    pix = np.zeros((n, n), dtype=np.uint8)
    length = indices[0].shape[0]
    arr_rep = np.tile(np.arange(0, 256).astype(np.uint8), (length//256+1)*256)[:length]
    pix[indices] = arr_rep

    img = Image.fromarray(pix)
    img.show()

    indexes = np.arange(0, length)

    for i in range(0, 100):
        print("i: {}".format(i))
        idxs = indexes*2+i
        arr_idx = base_num_arr[idxs]

        pix[:] = 0
        pix[indices] = arr_idx

        img = Image.fromarray(colors_rgb[pix].astype(np.uint8))
        img.save(dir_images+"pi_digits_base_2_nr_1_{:03}.png".format(i))

    # h = 550
    # for w in range(1, 600+1):
    #     print("w: {}".format(w))
    #     # w = 20
    #     arr_idx = base_num_arr[:h*w].reshape((h, w))

    #     img = Image.fromarray(colors_rgb[arr_idx].astype(np.uint8))
    #     # img_resize = img.resize((img.width*resize_factor, img.height*resize_factor))
    #     # img_resize.save(dir_images+"pi_digits_{:03}.png".format(w))
    #     img.save(dir_images+"pi_digits_base_2_{:03}.png".format(w))
