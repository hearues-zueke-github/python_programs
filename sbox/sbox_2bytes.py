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


def test_sbox_2bytes_nr_1():
    seed_sbox = b'HelloWorld!f234t89hnf98fj9ew8r4nfq948t7nw9tnaw94nt41e2EWR123r23u'
    
    sbox_2bytes = Sbox2Bytes(seed_sbox=seed_sbox)
    sbox_2bytes_2 = Sbox2Bytes(seed_sbox=seed_sbox)

    assert np.all(sbox_2bytes.arr_matrix == sbox_2bytes_2.arr_matrix)
    assert np.all(sbox_2bytes.sbox == sbox_2bytes_2.sbox)

    assert hashlib.sha512(sbox_2bytes.arr_matrix.tobytes()).hexdigest() == 'd5f397e19281eb2aefd0ab92b9212a56c8e255b1947b73fcfc70a2af5a047335ca4cc7cf611543024228651769537ba6365ba5384c730e53acdeab1c27997fdd'
    assert hashlib.sha512(sbox_2bytes.sbox.tobytes()).hexdigest() == '87069a30df84e27b6311a494df328649176362198a87a52b1f9a21e9a7958ecadd92b65fcdf6586835cac893d19586c5dca4b3e44d520ea480fe37379bffab9c'


def test_sbox_2bytes_all():
    test_sbox_2bytes_nr_1()


if __name__ == '__main__':
    seed_sbox = b'HelloWorld!f234t89hnf98fj9ew8r4nfq948t7nw9tnaw94nt41e2EWR123r23v'
    
    sbox_2bytes = Sbox2Bytes(seed_sbox=seed_sbox)

    fig, axs = plt.subplots(nrows=2, ncols=2)


    l_y_u_mean = [np.mean(d['u_diff']) for d in sbox_2bytes.l_d_histogram_stats]
    l_x_u_mean = list(range(1, len(l_y_u_mean) + 1))

    ax = axs[0][0]
    ax.plot(l_x_u_mean, l_y_u_mean, color='#0000FF', marker='o', linestyle='-')

    ax.set_title('Mean of the u_diff array values')


    l_y_c_mean = [np.mean(d['c_diff']) for d in sbox_2bytes.l_d_histogram_stats]
    l_x_c_mean = list(range(1, len(l_y_c_mean) + 1))

    ax = axs[1][0]
    ax.plot(l_x_c_mean, l_y_c_mean, color='#0000FF', marker='o', linestyle='-')

    ax.set_title('Mean of the c_diff array values')


    l_y_u_median = [np.median(d['u_diff']) for d in sbox_2bytes.l_d_histogram_stats]
    l_x_u_median = list(range(1, len(l_y_u_median) + 1))

    ax = axs[0][1]
    ax.plot(l_x_u_median, l_y_u_median, color='#0000FF', marker='o', linestyle='-')

    ax.set_title('Median of the u_diff array values')


    l_y_c_median = [np.median(d['c_diff']) for d in sbox_2bytes.l_d_histogram_stats]
    l_x_c_median = list(range(1, len(l_y_c_median) + 1))

    ax = axs[1][1]
    ax.plot(l_x_c_median, l_y_c_median, color='#0000FF', marker='o', linestyle='-')

    ax.set_title('Median of the c_diff array values')


    plt.show()
