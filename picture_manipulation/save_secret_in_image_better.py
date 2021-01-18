#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys
import string

import shutil

from typing import List, Dict, Set, Mapping, Any, Tuple

# import tempfile
from memory_tempfile import MemoryTempfile
tempfile : MemoryTempfile = MemoryTempfile()

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
CURRENT_DIR = os.getcwdb().decode('utf-8')

from PIL import Image

import numpy as np
import pandas as pd

sys.path.append('..')
import utils
from utils_multiprocessing_manager import MultiprocessingManager


def convert_1d_to_2d_arr(arr, length):
    arr_2d = np.zeros((arr.shape[0]-length+1, length), dtype=np.uint8)
    for i in range(0, length-1):
        arr_2d[:, i] = arr[i:-length+1+i]
    arr_2d[:, -1] = arr[length-1:]
    return arr_2d


lst_int_base_100 = string.printable
# lst_int_base_100 = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_.,:;!?#$%&()[]{}/\\ \'\"")
base_100_len = len(lst_int_base_100)
assert base_100_len == 100
dict_base_100_int = {v: i for i, v in enumerate(lst_int_base_100, 0)}

def convert_base_100_to_int(num_base_100):
    b = 1
    s = 0
    for i, v in enumerate(reversed(list(num_base_100)), 0):
        n = dict_base_100_int[v]
        s += n*b
        b *= base_100_len
    return s


def convert_int_to_base_100(num_int):
    l = []
    while num_int > 0:
        l.append(num_int % base_100_len)
        num_int //= base_100_len
    n = list(map(lambda x: lst_int_base_100[x], reversed(l)))
    return "".join(n)


def convert_int_to_lst_bin(num_int):
    return list(map(int, bin(num_int)[2:]))


def convert_lst_bin_to_int(l_bin):
    arr = np.array(l_bin, dtype=object)
    length = arr.shape[0]
    return np.sum(arr*2**np.arange(length-1, -1, -1).astype(object))


secret_test = "test123%$&/?!-_,:.;"

assert secret_test==convert_int_to_base_100(convert_base_100_to_int(secret_test))
assert 12345678901234567890==convert_base_100_to_int(convert_int_to_base_100(12345678901234567890))

assert 1234567==convert_lst_bin_to_int(convert_int_to_lst_bin(1234567))
assert [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]==convert_int_to_lst_bin(convert_lst_bin_to_int([1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]))


prefix_int : int = 0xabcd2533
suffix_int : int = 0x34bf4634

arr_prefix : np.ndarray = np.array(convert_int_to_lst_bin(prefix_int), dtype=np.uint8)
arr_suffix : np.ndarray = np.array(convert_int_to_lst_bin(suffix_int), dtype=np.uint8)

len_arr_prefix = arr_prefix.shape[0]
len_arr_suffix = arr_suffix.shape[0]

if __name__ == '__main__':
    print('Hello World!')

    path_images = os.path.join(PATH_ROOT_DIR, 'images/')
    assert os.path.exists(path_images)

    img_src_path : str = "images/orig_image_2_no_secret.png"
    img_src_new_path : str = "images/orig_image_2_no_secret_new.png"
    img_dst_path : str = "images/orig_image_2_with_secret.png"

    MAX_WIDTH = 200
    MAX_HEIGHT = 150

    # MAX_WIDTH = 400
    # MAX_HEIGHT = 300

    # MAX_WIDTH = 800
    # MAX_HEIGHT = 600

    img_src_orig : Image = Image.open(img_src_path)
    pix : np.ndarray = np.array(img_src_orig)
    if not os.path.exists(img_src_new_path):
        if len(pix.shape)==3:
            # remove alpha channel, if alpha is contained! also save the file again.
            if pix.shape[2]==4:
                pix = pix[..., :3]
        
        img2 : Image = Image.fromarray(pix)
        width, height = img2.size
        if width > MAX_WIDTH or height > MAX_HEIGHT:
            if width > MAX_WIDTH:
                width_new = MAX_WIDTH
                height_new = int(width_new * height / width)
            elif height > MAX_HEIGHT:
                height_new = MAX_HEIGHT
                width_new = int(height_new * width / height)

            img2 = img2.resize(size=(width_new, height_new), resample=Image.LANCZOS)

        img2.save(img_src_new_path)

    img_src : Image = Image.open(img_src_new_path)
    pix_orig : np.ndarray = np.array(img_src)

    assert len(pix_orig.shape) == 3
    assert pix_orig.shape[2] == 3

    shape_img_src : Tuple[int, int, int] = pix_orig.shape
    print("shape_img_src: {}".format(shape_img_src))

    pix : np.ndarray = pix_orig.copy()

    arr_1_bit : np.ndarray = (pix & 0x1).reshape((-1, ))
    arr_1_bit_orig : np.ndarray = arr_1_bit.copy()
    len_arr_1_bit : int = arr_1_bit.shape[0]

    l_secret_str = [
        'hello',
        'this is',
        'a little test! 123?',
        """def print_some_stuff():\n print(\"Test! 123=!=!=!= xD\")""",
        'lolololululululusfsfsdlfjsdlfjsdlfjsfjsfjsklfjksjfsjfsfjsdlfjafwefawoi',
        'lolololululululusfsfsdlfjsdlf',
        'lolololulululujsdlfjsfjsfjsklfjksjfsjfsfjsdlfjafwefawoi',
        'lolololulululujsdlfjsfjsfjsklfjksjfsjfsfjsdlfjafwefawoi'*3,
        'lolololulululujsdlfjsfjsfjsklfjksjfsjfsfjsdlfjafwefawoi'*6,
    ]

    # amount_secrets = 2
    # secret_len = 400
    # l_secret_str = [''.join(np.random.choice(lst_int_base_100, (secret_len, ))) for _ in range(0, amount_secrets)]

    MIN_BITS_LENGTH = 16
    JUMP_MIN = 10
    JUMP_MAX = 16

    def create_secret_bin_content(secret_str : str) -> List[int]:
        secret_int : List[int] = convert_base_100_to_int(secret_str)
        secret_bin : List[int] = convert_int_to_lst_bin(secret_int)
        l_len_secret_bin = convert_int_to_lst_bin(len(secret_bin))
        len_l_len_secret_bin = len(l_len_secret_bin)
        assert len_l_len_secret_bin <= MIN_BITS_LENGTH
        if len_l_len_secret_bin < MIN_BITS_LENGTH:
            l_len_secret_bin = [0] * (MIN_BITS_LENGTH - len_l_len_secret_bin) + l_len_secret_bin
        return arr_prefix.tolist() + secret_bin + l_len_secret_bin + arr_suffix.tolist()

    l_arr_secret_bin_content = [np.array(create_secret_bin_content(secret_str), dtype=np.uint8) for secret_str in l_secret_str]

    # TODO: make this into a multiprocessing function too!
    def find_best_possible_parameters(n : int=100):
        def inner_function():
            arr_line_param_offset : np.ndarray = np.random.randint(0, len_arr_1_bit, (len(l_secret_str), ))
            arr_line_param_jumps : np.ndarray = np.random.randint(JUMP_MIN, JUMP_MAX+1, (len(l_secret_str), ))
            l_arr_pos : List[np.ndarray] = [(np.arange(0, len(l_bin_content)) * jumps + offset) % len_arr_1_bit for l_bin_content, offset, jumps in zip(l_arr_secret_bin_content, arr_line_param_offset, arr_line_param_jumps)]

            # check, if any overlaps are there between the position of each secret!
            i_1 : int
            arr_secret_bin_content_1 : np.ndarray
            arr_pos_1 : np.ndarray
            for i_1, (arr_secret_bin_content_1, arr_pos_1) in enumerate(zip(l_arr_secret_bin_content[:-1], l_arr_pos[:-1]), 0):
                i_2 : int
                arr_secret_bin_content_2 : np.ndarray
                arr_pos_2 : np.ndarray
                for i_2, (arr_secret_bin_content_2, arr_pos_2) in enumerate(zip(l_arr_secret_bin_content[i_1+1:], l_arr_pos[i_1+1:]), i_1+1):
                    arr_idxs_bool_1 : np.ndarray = np.isin(arr_pos_1, arr_pos_2)
                    if np.any(arr_idxs_bool_1):
                        print("Some Equal postiions! i_1: {}, i_2: {}".format(i_1, i_2))
                        arr_idxs_bool_2 : np.ndarray = np.isin(arr_pos_2, arr_pos_1)
                        
                        arr_bin_1_part : np.ndarray = arr_secret_bin_content_1[arr_idxs_bool_1]
                        arr_bin_2_part : np.ndarray = arr_secret_bin_content_2[arr_idxs_bool_2]
                        if np.any(arr_bin_1_part != arr_bin_2_part):
                            print("arr_bin_1_part: {}".format(arr_bin_1_part))
                            print("arr_bin_2_part: {}".format(arr_bin_2_part))
                            return None

            return arr_line_param_offset, arr_line_param_jumps, l_arr_pos

        arr_line_param_offset = None
        arr_line_param_jumps = None
        l_arr_pos = None

        for nr_try in range(1, n + 1):
            ret = inner_function()
            if ret is None:
                print(f'Failed to find good params at nr_try {nr_try}!')
                continue
            print(f'Found params at nr_try {nr_try}!')
            arr_line_param_offset, arr_line_param_jumps, l_arr_pos = ret
            break

        return arr_line_param_offset, arr_line_param_jumps, l_arr_pos

    # TODO: make this multiprocessing possible!
    arr_line_param_offset, arr_line_param_jumps, l_arr_pos = find_best_possible_parameters(n=1000000)

    if arr_line_param_offset is None:
        sys.exit('Failed to find good params!')

    print("arr_line_param_offset: {}".format(arr_line_param_offset))
    print("arr_line_param_jumps: {}".format(arr_line_param_jumps))

    l_params = [(jump, offset, arr_secret_bin_content.shape[0]) for jump, offset, arr_secret_bin_content in zip(arr_line_param_jumps, arr_line_param_offset, l_arr_secret_bin_content)]

    # apply the bit changes to the pix array!
    for arr_pos, arr_secret_bin_content in zip(l_arr_pos, l_arr_secret_bin_content):
        arr_1_bit[arr_pos] = arr_secret_bin_content

    pix_secret = (pix & 0xF8) | arr_1_bit.reshape(pix.shape)

    pix_1_bit_orig = arr_1_bit_orig.reshape(shape_img_src) * 255
    pix_1_bit = arr_1_bit.reshape(shape_img_src) * 255

    Image.fromarray(pix_1_bit_orig).save('images/img_path_src_1bit_orig.png')
    Image.fromarray(pix_1_bit).save('images/img_path_src_1bit_encoded_in.png')

    img_secret : Image = Image.fromarray(pix_secret)
    img_secret.save(img_dst_path)

    
    img_src = Image.open(img_src_new_path)
    img_dst = Image.open(img_dst_path)

    pix_src = np.array(img_src)
    pix_dst = np.array(img_dst)

    pix_src_1bit = (pix_src & 0x1) * 255
    pix_dst_1bit = (pix_dst & 0x1) * 255
    pix_src_dst_1bit = pix_src_1bit ^ pix_dst_1bit

    img_path_src_1bit = 'images/img_path_src_1bit.png'
    img_path_dst_1bit = 'images/img_path_dst_1bit.png'
    img_path_src_dst_1bit = 'images/img_path_src_dst_1bit.png'

    Image.fromarray(pix_src_1bit).save(img_path_src_1bit)
    Image.fromarray(pix_dst_1bit).save(img_path_dst_1bit)
    Image.fromarray(pix_src_dst_1bit).save(img_path_src_dst_1bit)

    # try to find some matches!
    img_dst : Image = Image.open(img_dst_path)
    pix_dst : np.ndarray = np.array(img_dst)
    assert len(pix_dst.shape) == 3
    assert pix_dst.shape[2] == 3

    arr_dst_1_bit : np.ndarray = (pix_dst & 0x1).reshape((-1, ))

    def func_find_possible_params(
        arr_dst_1_bit : np.ndarray,
        arr_prefix : np.ndarray,
        arr_suffix : np.ndarray,
        l_jump : List[int],
    ) -> List[Tuple[int, int, int]]:
        len_arr_dst_1_bit : int = arr_dst_1_bit.shape[0]
        
        l_possible_params : List[Tuple[int, int, int]] = []

        len_arr_prefix : int = len(arr_prefix)
        len_arr_suffix : int = len(arr_suffix)

        xs_prefix_basic : np.ndarray = np.arange(0, len_arr_prefix)
        xs_suffix_basic : np.ndarray = np.arange(0, len_arr_suffix)
        for jump in l_jump:
        # for jump in range(JUMP_MIN, JUMP_MAX + 1):
            print("jump: {}".format(jump))
            xs_jump = (xs_prefix_basic * jump) % len_arr_dst_1_bit
            for offset_prefix in range(0, len_arr_dst_1_bit):
                xs_prefix = (xs_jump + offset_prefix) % len_arr_dst_1_bit
                # first: find the left part (prefix)
                if np.all(np.equal(arr_dst_1_bit[xs_prefix], arr_prefix)):
                    print("offset_prefix: {}".format(offset_prefix))
                    for offset_jump in range(MIN_BITS_LENGTH + len_arr_prefix, len_arr_dst_1_bit - len_arr_suffix):
                        offset_suffix = offset_jump * jump
                        # print("offset_suffix: {}".format(offset_suffix))
                        xs_suffix = (xs_suffix_basic * jump + offset_prefix + offset_suffix) % len_arr_dst_1_bit
                        # send: find the right part (suffix)
                        if np.all(np.equal(arr_dst_1_bit[xs_suffix], arr_suffix)):
                            arr_pos : np.ndarray = (np.arange(len_arr_prefix, offset_jump) * jump + offset_prefix) % len_arr_dst_1_bit
                            arr_part : np.ndarray = arr_dst_1_bit[arr_pos]

                            arr_secret_bin : np.ndarray = arr_part[:-MIN_BITS_LENGTH]
                            arr_secret_bin_len : np.ndarray = arr_part[-MIN_BITS_LENGTH:]

                            # third: check, if the content length is the same as the given binary number length!
                            if arr_secret_bin.shape[0] == convert_lst_bin_to_int(arr_secret_bin_len):                                
                                t_params = (jump, offset_prefix, offset_jump + len_arr_suffix)
                                print("t_params: {}".format(t_params))
                                l_possible_params.append(t_params)

        return l_possible_params

    mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())

    mult_proc_mng.test_worker_threads_response()

    l_jump_all = list(range(JUMP_MIN, JUMP_MAX+1))
    amount_parts = 7
    len_l_jump_all = len(l_jump_all)
    factor = len_l_jump_all / amount_parts
    l_jump_range_one = [0] + [int(factor * j) for j in range(1, amount_parts)] + [len_l_jump_all]
    l_l_jump = [l_jump_all[i1:i2] for i1, i2 in zip(l_jump_range_one[:-1], l_jump_range_one[1:])]

    print("l_l_jump: {}".format(l_l_jump))
    # del mult_proc_mng
    # sys.exit()

    print('Define new Function!')
    mult_proc_mng.define_new_func('find_possible_params', func_find_possible_params)
    print('Do the jobs!!')
    l_ret : List[List[Tuple[int, int, int]]] = mult_proc_mng.do_new_jobs(
        ['find_possible_params']*len(l_l_jump),
        [
            (arr_dst_1_bit, arr_prefix, arr_suffix, l_jump)
            for l_jump
            in l_l_jump
        ]
    )
    print("len(l_ret): {}".format(len(l_ret)))
    # print("l_ret: {}".format(l_ret))

    mult_proc_mng.test_worker_threads_response()
    del mult_proc_mng

    l_possible_params : List[Tuple[int, int, int]] = [t for l_possible_params_part in l_ret for t in l_possible_params_part]

    l_params = sorted(l_params)
    l_possible_params = sorted(l_possible_params)

    # only for checking, if the output is correct or not!
    assert l_params == l_possible_params

    print("l_possible_params:\n{}".format(l_possible_params))
    print("l_params:\n{}".format(l_params))

    print("len(l_possible_params): {}".format(len(l_possible_params)))
    print("len(l_params): {}".format(len(l_params)))
