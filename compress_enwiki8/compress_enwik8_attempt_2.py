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

import multiprocessing as mp

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
import global_object_getter_setter

import utils_compress_enwik8

from create_stats_enwik8 import calc_sorted_stats

def calc_stats_using_bytes_tuple(arr, max_len):
    used_len = 1000000
    max_amount_values = 50
    # max_len = 6
    l_stat = []
    s_chars = set()
    for pos_i in range(0, arr.shape[0], used_len):
        arr_1 = arr[pos_i:pos_i+used_len+max_len].reshape((-1, 1))
        print("pos_i: {:9}, {:9}".format(pos_i, pos_i+used_len))
        for _ in range(0, max_len-1):
            arr_1 = np.hstack((arr_1[:-1], arr_1[1:, -1:]))
        u, c = np.unique(arr_1.reshape((-1, )).view(','.join(['u1']*max_len)), return_counts=True)
        if max_len == 1:
            d = {tuple((t, )): j for t, j in zip(u, c)}
        else:
            d = {tuple(t): j for t, j in zip(u, c)}

        # get the max len for each seperate combined bytes!
        l_t, l_j = list(zip(*list(d.items())))
        i_max = np.argmax(l_j)

        print("- max_len: {:2}, amount: {:10}, mult: {:10}, t: {}".format(max_len, l_j[i_max], max_len*l_j[i_max], l_t[i_max]))
        print("-- len(d): {}".format(len(d)))
        s_chars |= set(list(d.keys()))

        l = list(d.items())
        l_sort = sorted(list(d.items()), reverse=True, key=lambda x: (x[1], x[0]))

        l_stat.append('{:9},{:9}:{}'.format(
            pos_i,
            pos_i+used_len,
            '|'.join(['{},{:5}'.format(''.join(map(lambda x: '{:02X}'.format(x), t)), c) for t, c in l_sort[:max_amount_values]])
        ))
    l = sorted(s_chars)
    print("l: {}".format(l))
    print("len(l): {}".format(len(l)))

    with open(TEMP_DIR+'enwik8_stats_max_len_{}.txt'.format(max_len), 'w') as f:
        f.write('\n'.join(l_stat)+'\n')


if __name__ == "__main__":
    file_object_name = 'global_compress_enwik8_attempt_2_object'

    if not global_object_getter_setter.do_object_exist(file_object_name):
        arr = utils_compress_enwik8.get_arr(used_length=-1)


        # calc_stats_using_bytes_tuple(arr, 1)
        
        # l_proc = []
        # cpu_count = mp.cpu_count()
        # for i in range(2, cpu_count+2):
        #     l_proc.append(mp.Process(target=calc_stats_using_bytes_tuple, args=(arr, i)))
        # for proc in l_proc: proc.start()
        # for proc in l_proc: proc.join()

        d_all_part, l_sort = calc_sorted_stats()
        d3 = d_all_part[3]
        l_k_3 = list(d3.keys())
        l_k_i_2_byte = [(k, (0, i)) for i, k in enumerate(l_k_3, 0)]
        
        l_sort_ge_4_byte = sorted([(len(k)*v, -len(k), v, k) for k1 in range(4, 14) for k, v in d_all_part[k1].items()], reverse=True)
        l_k_i_2_byte += [(k, (0, i)) for i, (_, _, _, k) in enumerate(l_sort_ge_4_byte[:256-len(l_k_3)], len(l_k_3))]

        l_k_i_3_byte = [(k, (0, i//256, i%256)) for i, (_, _, _, k) in enumerate(l_sort_ge_4_byte[256-len(l_k_3):], 0)]


        d_obj = {
            'arr': arr,
            'd_all_part': d_all_part,
            'l_sort': l_sort,
            'd3': d3,
            'l_k_3': l_k_3,
            'l_k_i_2_byte': l_k_i_2_byte,
            'l_sort_ge_4_byte': l_sort_ge_4_byte,
            'l_k_i_2_byte': l_k_i_2_byte,
            'l_k_i_3_byte': l_k_i_3_byte,
            # 'd_k_to_count': d_k_to_count,
            # 'd_k_to_i_byte': d_k_to_i_byte,
            # 'l_arr': l_arr,
        }
        print('Save global DATA!')
        global_object_getter_setter.save_object(file_object_name, d_obj)
    else:
        print('Load global DATA!')
        d_obj = global_object_getter_setter.load_object(file_object_name)
        arr = d_obj['arr']
        d_all_part = d_obj['d_all_part']
        l_sort = d_obj['l_sort']
        d3 = d_obj['d3']
        l_k_3 = d_obj['l_k_3']
        l_k_i_2_byte = d_obj['l_k_i_2_byte']
        l_sort_ge_4_byte = d_obj['l_sort_ge_4_byte']
        l_k_i_2_byte = d_obj['l_k_i_2_byte']
        l_k_i_3_byte = d_obj['l_k_i_3_byte']
        # d_k_to_count = d_obj['d_k_to_count']
        # d_k_to_i_byte = d_obj['d_k_to_i_byte']
        # l_arr = d_obj['l_arr']
    
    d_k_to_count = {k: v for k1 in range(3, 14) for k, v in d_all_part[k1].items()}
    d_k_to_i_byte = dict(l_k_i_2_byte+l_k_i_3_byte)

    print("len(d_k_to_count): {}".format(len(d_k_to_count)))
    print("len(d_k_to_i_byte): {}".format(len(d_k_to_i_byte)))

    assert set(list(d_k_to_count)) == set(list(d_k_to_i_byte))

    l_arr = arr.tolist()
    l_encrypt = []

    max_len = 13
    length = len(l_arr)
    i = 0
    while i < length:
        l = []
        l_count = []
        l_mult = []

        length_byte = 3
        j = i+3
        while j <= length and length_byte <= max_len:
            t = tuple(l_arr[i:j])
            
            if t in d_k_to_count:
                l.append(t)
                c = d_k_to_count[t]
                l_count.append(c)
                l_mult.append(len(t)*c)

            j += 1
            length_byte += 1

        if len(l) == 0:
            l_encrypt.append(l_arr[i])
            i += 1
        else:
            i_max = len(l)-1
            # i_max = np.argmax(l_mult)
            t_max = l[i_max]
            l_byte = d_k_to_i_byte[t_max]
            l_encrypt.extend(l_byte)
            i += len(t_max)

            # print("l: {}".format(l))
            # print("l_count: {}".format(l_count))
            # print("l_mult: {}".format(l_mult))
            # break

        if i % 10000 == 0:
            print("i: {}".format(i))

        if i > 10000000:
            break
