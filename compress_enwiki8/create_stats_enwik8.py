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

from os.path import expanduser

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
import global_object_getter_setter


def calc_sorted_stats():
    file_path_template = PATH_ROOT_DIR+'data_enwik8/enwik8_stats_max_len_{}.txt'
    # file_path_template = TEMP_DIR+'enwik8_stats_max_len_{}.txt'

    d_all = {}
    d_all_part = {}
    for i in range(2, 14):
        print("i: {}".format(i))
        file_path = file_path_template.format(i)
        with open(file_path, 'r') as f:
            l_line = f.read().rstrip('\n').split('\n')

        l_line = [[t.split(',') for t in l.split(':')[1].split('|')] for l in l_line]
        l_line = [[[tuple(int(t_str[i:i+2], 16) for i in range(0, len(t_str), 2)), int(c_str)] for t_str, c_str in l] for l in l_line]

        l_d = [{k: v for k, v in l} for l in l_line]
        d = l_d[0]
        for d_next in l_d[1:]:
            for k, v in d_next.items():
                if k in d:
                    d[k] += v
                else:
                    d[k] = v

        for k, v in d.items():
            assert k not in d_all
            d_all[k] = v
        d_all_part[i] = d

    l_sort = sorted([(len(k)*v, -len(k), v, k) for k, v in d_all.items() if len(k) > 2], reverse=True)
    return d_all_part, l_sort


if __name__ == "__main__":
    d_all_part, l_sort = calc_sorted_stats()    
    print('\n'.join([str(l) for l in l_sort[:30]]))
