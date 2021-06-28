#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9.5 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import hashlib
import os
import pdb
import re
import sys
import traceback

import numpy as np
import pandas as pd
import multiprocessing as mp

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union
from PIL import Image

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()

import importlib.util as imp_util

spec = imp_util.spec_from_file_location("utils", os.path.join(PATH_ROOT_DIR, "../utils.py"))
utils = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils)

mkdirs = utils.mkdirs

spec = imp_util.spec_from_file_location("utils_multiprocessing_manager", os.path.join(PATH_ROOT_DIR, "../utils_multiprocessing_manager.py"))
utils_multiprocessing_manager = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils_multiprocessing_manager)

MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

ITERATIONS_ROTATE_FLIP_SHIFT_ADD = 16

class Sbox2Bytes(Exception):
    def __init__(self, seed_sbox):
        if len(seed_sbox) >= 256:
            seed_sbox = seed_sbox[:256]
        else:
            seed_sbox = seed_sbox + b'\x00' * (256 - len(seed_sbox))

        self.seed_sbox = seed_sbox
        print("self.seed_sbox: {}".format(self.seed_sbox))

        matrix_size = 32
        arr_matrix = np.zeros((matrix_size, matrix_size), dtype=np.uint8)
        arr_matrix[:, :] = (np.arange(0, matrix_size) + 1).reshape((-1, 1))
        self.arr_matrix = arr_matrix

        arr_op_num_bit_all, sbox = self.generate_init_sbox(
            seed_sbox=seed_sbox,
            arr_matrix=arr_matrix,
            f_arr=self.f_mix_arr_inplace,
        )

        self.arr_op_num_bit_all = arr_op_num_bit_all
        self.sbox = sbox

        assert np.unique(sbox).shape[0] == 2**16

        print("arr_matrix:\n{}".format(arr_matrix))

    def f_mix_arr_inplace(self, arr_matrix: np.ndarray) -> None:
        arr1 = (arr_matrix ^ np.roll(np.flip(arr_matrix.T, 1), 1, 0))
        arr2 = (arr_matrix ^ np.roll(np.flip(arr_matrix.T, 0), 1, 1))
        arr_matrix[:] = (arr1 + arr2)


    def generate_init_sbox(self, seed_sbox, arr_matrix, f_arr):
        sbox = np.arange(0, 2**16).astype(np.uint16)
        sbox = np.roll(sbox, 1)

        arr_op_num_byte_all = np.array(list(seed_sbox), dtype=np.uint8)
        arr_op_num_bit_all = np.array([(b >> i) & 0x1 for b in arr_op_num_byte_all for i in range(7, -1, -1)], dtype=np.uint8)
        print("arr_op_num_bit_all: {}".format(arr_op_num_bit_all))

        arr_op_num_byte_row = arr_op_num_bit_all.reshape((-1, arr_matrix.shape[1]))

        def apply_permutation_on_the_sbox(arr_idx, arr_matrix, sbox):
            arr_idx_num = np.cumsum(arr_matrix.view(np.uint16)+1).astype(np.uint16)

            arr_idx[:] = 0

            idx_num_order = 0
            d_idx_num = {}
            for idx_num in arr_idx_num:
                while arr_idx[idx_num] == 1 or idx_num in d_idx_num:
                    idx_num = (idx_num + 1) % 2**16
                arr_idx[idx_num] = 1
                d_idx_num[idx_num] = idx_num_order
                idx_num_order += 1

            l_idx_num = [idx_num for idx_num, _ in sorted(d_idx_num.items(), key=lambda x: (x[1], ))]
            arr_idx_num_to = np.array(l_idx_num)
            arr_idx_num_from = np.roll(arr_idx_num_to, 1)
            sbox[arr_idx_num_to] = sbox[arr_idx_num_from]

            arr_idx_to = np.sort(np.where(
                (sbox == np.arange(0, 2**16)) |
                (sbox == np.roll(np.arange(0, 2**16), 1)) |
                (sbox == np.roll(np.arange(0, 2**16), -1))
            )[0])
            arr_idx_from = (np.arange(0, len(arr_idx_to)) - 2) % 2**16
            for idx_from, idx_to in zip(arr_idx_from, arr_idx_to):
                sbox[idx_from], sbox[idx_to] = sbox[idx_to], sbox[idx_from]

        arr_idx = np.zeros((2**16, ), dtype=np.uint16)
        arr_idx_from = np.roll(np.arange(0, arr_matrix.shape[1]), 1)
        arr_idx_to = np.roll(np.arange(0, arr_matrix.shape[1]), 0)

        f_arr(arr_matrix)
        for arr_op_num_byte in arr_op_num_byte_row:
            for idx_from, idx_to, op_num in zip(arr_idx_from, arr_idx_to, arr_op_num_byte):
                arr_hash = np.array(list(hashlib.sha256(arr_matrix[idx_from].tobytes()).digest()), dtype=np.uint8)

                if op_num == 0:
                    arr_matrix[idx_to] += arr_hash
                elif op_num == 1:
                    arr_matrix[idx_to] ^= arr_hash
                else:
                    assert False

            apply_permutation_on_the_sbox(arr_idx=arr_idx, arr_matrix=arr_matrix, sbox=sbox)
            for _ in range(0, ITERATIONS_ROTATE_FLIP_SHIFT_ADD):
                f_arr(arr_matrix)
                apply_permutation_on_the_sbox(arr_idx=arr_idx, arr_matrix=arr_matrix, sbox=sbox)

        return arr_op_num_bit_all, sbox


if __name__ == '__main__':
    seed_sbox = b'HelloWorld!' * 16
    
    sbox_2bytes = Sbox2Bytes(seed_sbox=seed_sbox)
