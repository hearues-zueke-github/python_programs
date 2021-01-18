#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import io
import sys
import traceback

import numpy as np
import pandas as pd

from typing import List, Dict, Any, Callable, Set

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from pathlib import Path
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt

sys.path.append('..')
from utils import mkdirs

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

def main(d_env: Dict[str, Any]) -> None:
    def load_obj_in_d_env(obj_name: str, func: Callable[[Dict[str, Any]], Any], d_env: Dict[str, Any]) -> None:
        file_path_obj = OBJS_DIR_PATH + obj_name + '.pkl.gz'

        if not os.path.exists(file_path_obj):
            print("Creating '{}' object.".format(obj_name))

            obj = func(d_env)

            with gzip.open(file_path_obj, 'wb') as f:
                dill.dump(obj, f)
        else:
            print("Loading '{}' object.".format(obj_name))
            with gzip.open(file_path_obj, 'rb') as f:
                obj = dill.load(f)

        d_env[obj_name] = obj

    def func_obj_d_chunck_size_d_xy(d_env: Dict[str, Any]) -> Dict[int, Any]:
        return {}

    d_chunck_size_d_xy = func_obj_d_chunck_size_d_xy(d_env)

    d_env['d_chunck_size_d_xy'] = d_chunck_size_d_xy

    # for 

    # def func_obj_d_df(d_env: Dict[str, Any]) -> Any:
    #     d_wb_d_df = d_env['d_wb_d_df']
    #     df_merge = create_df_merge_from_basal_bolus(d_wb_d_df)
    #     d_df = create_d_df_base(d_wb_d_df, df_merge)

    # return d_df


def base_convert_b1_to_b2(l1, b1, b2):
    n = 0
    p1 = 1
    for v in l1[::-1]:
        n += v * p1
        p1 *= b1

    l2 = []
    while n > 0:
        l2.append(n % b2)
        n //= b2

    return l2[::-1]


if __name__ == '__main__':
    # d_env: Dict[str, Any] = {}
    # main(d_env=d_env)

    sys.exit()

    # # b1 = io.BytesIO()
    # # b2 = open(b1, 'wb')
    # b1 = MemoryTempfile().TemporaryFile()
    # # with tempfile.TemporaryFile() as tf:
    # with gzip.open(b1, 'wb', compresslevel=9) as f:
    #     f.write(b'Test123!')

    # b1.seek(0)

    # with gzip.open(b1, 'rb') as f:
    #     content = f.read()
    # print("content: {}".format(content))

    # sys.exit()

    # print("Hello World!")

    tmp_folder = TEMP_DIR + 'content_tpl/'
    mkdirs(tmp_folder)

    byte_length = 1

    # file_path_content_tpl = tmp_folder + 'content_tpl_byte_len_{}.pkl.gz'.format(byte_length)
    file_path_l_sorted = tmp_folder + 'l_sorted_byte_len_{}.pkl.gz'.format(byte_length)

    # if os.path.exists(file_path_l_sorted):
    # # if os.path.exists(file_path_content_tpl) and os.path.exists(file_path_l_sorted):
    #     sys.exit()

    with open(HOME_DIR+'Downloads/enwik8', 'rb') as f:
        content = f.read()

    l_column = ['chunck_size', 'folder_size_content']
    d_stats = {s: [] for s in l_column}

    content = content
    # content = content[:1000000]
    content_tpl = tuple([int(i) for i in content])
    # content = tuple([int(i) for i in content[:100000000]])
    # content = content[:1000000]

    d = defaultdict(int)
    for i in range(0, len(content_tpl)-byte_length+1):
        d[content_tpl[i:i+byte_length]] += 1

    l_count_hexstr = [(v, ''.join(["{:02X}".format(i) for i in k]), ''.join([chr(i) for i in k])) for k, v in d.items()]
    l_sorted = sorted(l_count_hexstr, reverse=False)
    print("l_sorted:")
    pprint(l_sorted[-200:])

    # with gzip.open(file_path_content_tpl, 'wb') as f:
    #     dill.dump(content_tpl, f)

    with gzip.open(file_path_l_sorted, 'wb') as f:
        dill.dump(l_sorted, f)

    sys.exit()


    # tmp_folder_test_compressions = TEMP_DIR + 'test_compressions/'
    # mkdirs(tmp_folder_test_compressions)

    l_chunck_size = [
        # 1000, 2000, 5000,
        # 10000, 20000, 50000,
        100000, 200000, 500000,
        # 1000000, 2000000, 5000000,
    ]

    d_chunck_size_d_xy = {}

    chunck_jump = 1000
    for chunck_size in l_chunck_size:
        l_x = []
        l_y = []
        d_chunck_size_xy[chunck_size] = {
            'x': l_x,
            'y': l_y,
        }
        for idx, pos in enumerate(range(0, len(content)-chunck_size+1, chunck_jump), 0):
            print("chunck_size: {}, pos: {}".format(chunck_size, pos))
            content_part = content[pos:pos+chunck_size]
            assert len(content_part) == chunck_size

            f_out = MemoryTempfile().TemporaryFile()
            with gzip.open(f_out, mode='wb', compresslevel=9) as f:
                f.write(content_part)
            f_out.seek(0, os.SEEK_END)
            l_x.append(pos)
            l_y.append(f_out.tell())
            # folder_size_content += f_out.tell()
            del f_out

    fig, axs = plt.subplots(figsize=(15, 9), nrows=3, ncols=1)
    fig.suptitle('Plot for diff chunck size compressions', fontsize=14)

    for i, chunck_size in enumerate(l_chunck_size, 0):
        ax = axs[i]
        d_xy = d_chunck_size_xy[chunck_size]
        l_x = d_xy['x']
        l_y = d_xy['y']
        ax.plot(l_x, l_y, marker='o', ms=5, color='#0000FFFF')
        ax.set_title('chunck_size: {}'.format(chunck_size))
        ax.set_ylim([0, chunck_size])
        ax.set_xlabel('Left position of chunck')
        ax.set_ylabel('Bytes')
    plt.tight_layout()

    fig, axs = plt.subplots(figsize=(15, 9), nrows=3, ncols=1)
    fig.suptitle('Plot for diff chunck size compressions (minimal points)', fontsize=14)

    for i, chunck_size in enumerate(l_chunck_size, 0):
        ax = axs[i]
        d_xy = d_chunck_size_xy[chunck_size]
        arr_x = np.array(d_xy['x'])
        arr_y = np.array(d_xy['y'])

        a1 = arr_y[:-2]
        a2 = arr_y[1:-1]
        a3 = arr_y[2:]

        idxs = np.hstack(((False, ), (a2 < a1) & (a2 < a3), (False, )))
        arr_x = arr_x[idxs]
        arr_y = arr_y[idxs]

        ax.plot(arr_x, arr_y, marker='o', ms=5, color='#0000FFFF')
        ax.set_title('chunck_size: {}'.format(chunck_size))
        ax.set_ylim([0, chunck_size])
        ax.set_xlabel('Left position of chunck')
        ax.set_ylabel('Bytes')
    plt.tight_layout()

    plt.show()

    sys.exit(0)

    for chunck_size in l_chunck_size:
    # chunck_size = 1000
        folder_size_content = 0
        # tmp_folder_chuncks = tmp_folder_test_compressions + 'chuncks_bytes_{:08}/'.format(chunck_size)
        # mkdirs(tmp_folder_chuncks)
        for idx, pos in enumerate(range(0, len(content), chunck_size), 0):
            print("idx: {}, pos: {}".format(idx, pos))
            content_part = content[pos:pos+chunck_size]
            # with gzip.open(tmp_folder_chuncks + 'part_{:06}.txt.gz'.format(idx), mode='wb', compresslevel=9) as f:
            #     f.write(content_part)
            
            # b1 = MemoryTempfile().TemporaryFile()
            # # with tempfile.TemporaryFile() as tf:
            # with gzip.open(b1, 'wb', compresslevel=9) as f:
            #     f.write(b'Test123!')

            # b1.seek(0)
            # with gzip.open(b1, 'rb') as f:
            #     content = f.read()
            # print("content: {}".format(content))

            f_out = MemoryTempfile().TemporaryFile()
            with gzip.open(f_out, mode='wb', compresslevel=9) as f:
                f.write(content_part)
            f_out.seek(0, os.SEEK_END)
            folder_size_content += f_out.tell()
            del f_out

            # with io.BytesIO() as f_out:
            #     with gzip.open(f_out, mode='wb', compresslevel=9) as f:
            #         f.write(content_part)
            #     f_out.seek(0, os.SEEK_END)
            #     folder_size_content += f_out.tell()

        # def get_file_size(path):
        #     with open(path, 'rb') as f:
        #         size = len(f.read())
        #     return size
        # folder_size_content = sum([get_file_size(os.path.join(r, f)) for r, ds, fs in os.walk(tmp_folder_chuncks) for f in fs])
        # folder_size_content = sum([Path(os.path.join(r, f)).stat().st_size for r, ds, fs in os.walk(tmp_folder_chuncks) for f in fs])
        print("folder_size_content: {}".format(folder_size_content))

        d_stats['chunck_size'].append(chunck_size)
        d_stats['folder_size_content'].append(folder_size_content)

    df = pd.DataFrame(data=d_stats, columns=l_column)
    print("df:\n{}".format(df))

    # {'chunck_size': [1000, 2000, 5000, 10000], 'folder_size_content': [560052, 499507, 448302, 420957]}

# with io.BytesIO:
#    chunck_size  folder_size_content
# 0         1000               544052
# 1         2000               491507
# 2         5000               445102
# 3        10000               419357

# with get_file_size:
#    chunck_size  folder_size_content
# 0         1000               560052
# 1         2000               499507
# 2         5000               448302
# 3        10000               420957


