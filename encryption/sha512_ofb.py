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

import matplotlib.pyplot as plt

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

class Sbox2Bytes(Exception):
    def __init__(self, seed_sbox):
        if len(seed_sbox) >= 64:
            seed_sbox = seed_sbox[:64]
        else:
            seed_sbox = seed_sbox + b'\x00' * (64 - len(seed_sbox))

        self.seed_sbox = seed_sbox
        print("self.seed_sbox: {}".format(self.seed_sbox))

        matrix_size = 64
        arr_matrix = np.zeros((matrix_size, matrix_size), dtype=np.uint8)
        arr_matrix[:, :] = (np.arange(0, matrix_size) + 1).reshape((-1, 1))
        self.arr_matrix = arr_matrix

        sbox = self.generate_init_sbox(
            seed_sbox=seed_sbox,
            arr_matrix=arr_matrix,
        )

        self.sbox = sbox

        assert np.unique(sbox).shape[0] == 2**16

        print("arr_matrix:\n{}".format(arr_matrix))

    def generate_init_sbox(self, seed_sbox, arr_matrix):
        sbox = np.arange(0, 2**16).astype(np.uint16)

        arr_seed_sbox_byte = np.array(list(seed_sbox), dtype=np.uint8)

        self.arr_seed_sbox_byte = arr_seed_sbox_byte
        arr_matrix[0] = arr_seed_sbox_byte

        arr_idx_total = np.zeros((2**16, ), dtype=np.uint16)
        arr_idx = np.zeros((2**16, ), dtype=np.uint16)

        arr_idx_from = np.arange(0, arr_matrix.shape[1])
        arr_idx_to = np.roll(arr_idx_from, -1)

        arr_idx_pos_cumsum = np.zeros((2**4, 2**12), dtype=np.uint16)

        l_d_histogram_stats = []
        self.l_d_histogram_stats = l_d_histogram_stats

        for iterations in range(1, 16+1):
            print("iterations: {}".format(iterations))

            for iters in range(0, 16):
                arr_matrix[:] = np.flip(arr_matrix.T, 1)
                
                for idx_from, idx_to in zip(arr_idx_from, arr_idx_to):
                    arr_hash = np.array(list(hashlib.sha512(arr_matrix[idx_from].tobytes()).digest()), dtype=np.uint8)

                    arr_matrix[idx_to] ^= arr_hash

                arr_idx_pos_cumsum[iters] = np.cumsum(arr_matrix.reshape((-1, )) + 1)

            arr_idx_pos_cumsum_all = (arr_idx_pos_cumsum + np.cumsum(arr_idx_pos_cumsum[:, -1]).astype(np.uint16).reshape((-1, 1)) - arr_idx_pos_cumsum[0, -1]).reshape((-1, ))

            for amount in range(2**16, 0, -1):
                idx = arr_idx_pos_cumsum_all[amount - 1] % amount
                if idx != amount - 1:
                    sbox[idx], sbox[amount - 1] = sbox[amount - 1], sbox[idx]

            assert np.unique(sbox).shape[0] == 2**16

            u, c = np.unique(np.abs(np.diff(sbox.astype(np.int64))), return_counts=True)
            l_d_histogram_stats.append(dict(
                iterations=iterations,
                sbox=sbox.copy(),
                u_diff=u,
                c_diff=c,
            ))

        return sbox


if __name__ == '__main__':
    seed_sbox = b'HelloWorld!f234t89hnf98fj9ew8r4nfq948t7nw9tnaw94nt41e2EWR123r23v'
    
    sbox_2bytes = Sbox2Bytes(seed_sbox=seed_sbox)
