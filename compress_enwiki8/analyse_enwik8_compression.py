#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import itertools
import pdb
import re
import sys
import traceback

import numpy as np
import pandas as pd

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from collections import defaultdict
from pprint import pprint

sys.path.append('..')
from utils import mkdirs

sys.path.append('../combinatorics')
from different_combinations import get_all_combinations_repeat

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)


def int_to_bin_str(n, bits):
    return bin(n)[2:].zfill(bits)

def bin_str_to_int_tpl(bin_str):
    return tuple(map(int, bin_str))


if __name__ == '__main__':
    # # try an other byte string compression example!
    # content = b'The data is UTF-8 clean. All characters are in the range U\'0000 to U\'10FFFF with valid encodings of 1 to 4 bytes. The byte values 0xC0, 0xC1, and 0xF5-0xFF never occur. Also, in the Wikipedia dumps, there are no control characters in the range 0x00-0x1F except for 0x09 (tab) and 0x0A (linefeed). Linebreaks occur only on paragraph boundaries, so they always have a semantic purpose. In the example below, lines were broken at 80 characters, but in reality each paragraph is one long line.'

    # t_content = tuple(map(int, content))
    # print("t_content: {}".format(t_content))
    # max_chunck_size = 7

    # d = defaultdict(list)
    # for chunck_size in range(1, max_chunck_size + 1):
    #     print("chunck_size: {}".format(chunck_size))
    #     # chunck_size = 3
    #     for i in range(0, len(t_content)-chunck_size+1):
    #         if i % 100000 == 0:
    #             print("i: {}".format(i))
    #         d[t_content[i:i+chunck_size]].append(i)

    # l_sorted = sorted([(len(v)*len(k), len(k), len(v), k) for k, v in d.items()])
    # print('l_sorted:')
    # pprint(l_sorted)

    # d_ranking = {k: rank for rank, _, _, k in l_sorted}

    # # algorithm
    # l_chuncks = []
    # length = len(t_content)
    # i = 0
    # while i < length:
    #     l = []
    #     for chunck_size in range(1, max_chunck_size + 1):
    #         t = t_content[i:i + chunck_size]
    #         l.append((d_ranking[t], len(t), t))
    #     l.sort()
    #     t = l[-1][-1]

    #     l_chuncks.append(t)
    #     i += len(t)

    # d_ranking_new = defaultdict(int)
    # for chunck in l_chuncks:
    #     d_ranking_new[chunck] += 1
    
    # l_choosen_chuncks = [(amount, -len(chunck), chunck) for chunck, amount in d_ranking_new.items()]
    # l_choosen_chuncks.sort(reverse=True)

    # l_word = [word for _, _, word in l_choosen_chuncks]

    # print("l_choosen_chuncks:")
    # pprint(l_choosen_chuncks)

    # # find the x! for the needed index table converting!
    # s = 0
    # x = 0
    # amount_chuncks = len(d_ranking_new)
    # while s < amount_chuncks:
    #     x += 2
    #     s += 2**x
    # print("x: {}".format(x))

    # # create index table for 2 bits, 4 bits, 6 bits, etc.
    # l_idx = []
    # arr_simple_prefix = get_all_combinations_repeat(m=2, n=2)
    # t_prefix = ()
    # for i in range(0, x // 2):
    #     if i > 0 and i % 3 == 0:
    #         t_prefix += tuple(arr_simple_prefix[3])
    #     t_prefix_new = t_prefix + tuple(arr_simple_prefix[i % 3])
    #     arr = get_all_combinations_repeat(m=2, n=(i+1)*2)
    #     l_idx.extend([t_prefix_new+tuple(v) for v in arr])

    # d_word_to_idx = {word: idx for word, idx in zip(l_word, l_idx)}

    # l_compressed_idx = [d_word_to_idx[word] for word in l_word]
    # l_compressed_idx_concat = [v for v in itertools.chain(*l_compressed_idx)]
    # print("l_compressed_idx_concat: {}".format(l_compressed_idx_concat))
    
    # d_chunck_words = defaultdict(list)
    # for word in l_word:
    #     d_chunck_words[len(word)].append(word)

    # for l in d_chunck_words.values():
    #     l.sort()

    # # amount_chunck_sizes = len(d_chunck_words.keys())

    # # # add the variable for the byte amount of a word
    # # # add the variable for the byte amount of a word

    # d_amount_word_byte_to_bit_field = {
    #      1: (0, 0) + (0, ),
    #      2: (0, 0) + (1, ),
    #      3: (0, 1) + (0, 0),
    #      4: (0, 1) + (0, 1),
    #      5: (0, 1) + (1, 0),
    #      6: (0, 1) + (1, 1),
    #      7: (1, 0) + (0, 0, 0),
    #      8: (1, 0) + (0, 0, 1),
    #      9: (1, 0) + (0, 1, 0),
    #     10: (1, 0) + (0, 1, 1),
    #     11: (1, 0) + (1, 0, 0),
    #     12: (1, 0) + (1, 0, 1),
    #     13: (1, 0) + (1, 1, 0),
    #     14: (1, 0) + (1, 1, 1),
    #     15: (1, 1) + (0, 0, 0, 0),
    #     16: (1, 1) + (0, 0, 0, 1),
    #     17: (1, 1) + (0, 0, 1, 0),
    #     18: (1, 1) + (0, 0, 1, 1),
    #     19: (1, 1) + (0, 1, 0, 0),
    #     20: (1, 1) + (0, 1, 0, 1),
    #     21: (1, 1) + (0, 1, 1, 0),
    #     22: (1, 1) + (0, 1, 1, 1),
    #     23: (1, 1) + (1, 0, 0, 0),
    #     24: (1, 1) + (1, 0, 0, 1),
    #     25: (1, 1) + (1, 0, 1, 0),
    #     26: (1, 1) + (1, 0, 1, 1),
    #     27: (1, 1) + (1, 1, 0, 0),
    #     28: (1, 1) + (1, 1, 0, 1),
    #     29: (1, 1) + (1, 1, 1, 0),
    #     30: (1, 1) + (1, 1, 1, 1),
    # }

    # # t_first_part = [
    # #     ('amount_chunck_sizes', bin_str_to_int_tpl(bin_str=int_to_bin_str(n=amount_chunck_sizes, bits=4))),
    # # ]# +
    # # # [
    # # #     [
    # # #         ('byte_length_{}'.format(i), d_amount_word_byte_to_bit_field[chunck_size]),
    # # #         ('length_arr_{}'.format(i), bin_str_to_int_tpl(bin_str=int_to_bin_str(n=len(l_word), bits=8))),
    # # #     ]
    # # #     for i, (chunck_size, l_word) in enumerate(d_chunck_words.items(), 0)
    # # # ]

    # amount_word = len(l_word)
    # t_amount_word = bin_str_to_int_tpl(bin_str=int_to_bin_str(n=amount_word, bits=0))
    # amount_bit_amount_word = len(t_amount_word)
    # t_amount_bit_amount_word = bin_str_to_int_tpl(bin_str=int_to_bin_str(n=amount_bit_amount_word, bits=6))
    
    # l_t_length_bytes_word = [d_amount_word_byte_to_bit_field[len(word)] for word in l_word]

    # t_first_part = [
    #     ('amount_bit_amount_word', t_amount_bit_amount_word),
    #     ('amount_word', t_amount_word),
    #     ('length_bytes_word', l_t_length_bytes_word),
    #     ('bytes_word', l_word),
    #     (),
    # ]

    # print('t_first_part:')
    # pprint(t_first_part)

    # sys.exit()

    print("Hello World!")

    with open("/home/doublepmcl/Downloads/enwik8", 'rb') as f:
        content = f.read()

    # content = tuple(map(int, content))
    content_size = 1000000
    # content_size = 10000000
    # content_size = 100000000

    t_content = tuple(map(int, content[:content_size]))

    d_chunck_size_d_word_count = defaultdict(lambda: defaultdict(int))

    folder_path = TEMP_DIR + 'd_word_count/'
    mkdirs(folder_path)
    
    length = len(t_content)
    max_chunck_size = 8
    for chunck_size in range(1, max_chunck_size+1):
        file_name = 'chunck_size_{}_length_{}.pkl.gz'.format(chunck_size, length)
        obj_path = folder_path + file_name
        if not os.path.exists(obj_path):
            print("Creating object: obj_path: {}".format(obj_path))
            d = d_chunck_size_d_word_count[chunck_size]
            for i in range(0, length - chunck_size + 1):
                if i % 100000 == 0:
                    print("chunck_size: {}, i: {}".format(chunck_size, i))
                d[t_content[i:i+chunck_size]] += 1
            print("- Saving object: obj_path: {}".format(obj_path))
            with gzip.open(obj_path, 'wb') as f:
                dill.dump(d, f)
        else:
            print("Loading object: obj_path: {}".format(obj_path))
            with gzip.open(obj_path, 'rb') as f:
                d_chunck_size_d_word_count[chunck_size] = dill.load(f)

    # set_word_1_byte_all = set(d_chunck_size_d_word_count[1].keys())
    # set_word_2_byte_all = set([t1+t2 for t1 in set_word_1_byte_all for t2 in set_word_1_byte_all])
    # set_word_3_byte_all = set([t1+t2 for t1 in set_word_2_byte_all for t2 in set_word_1_byte_all])

    len_set_word_1_byte_all = len(set(d_chunck_size_d_word_count[1].keys()))
    len_set_word_2_byte_all = len_set_word_1_byte_all**2
    len_set_word_3_byte_all = len_set_word_1_byte_all**3
    len_set_word_4_byte_all = len_set_word_1_byte_all**4
    len_set_word_5_byte_all = len_set_word_1_byte_all**5
    len_set_word_6_byte_all = len_set_word_1_byte_all**6

    set_word_1_byte_comb = set(d_chunck_size_d_word_count[1].keys())
    set_word_2_byte_comb = set(d_chunck_size_d_word_count[2].keys())
    set_word_3_byte_comb = set(d_chunck_size_d_word_count[3].keys())
    set_word_4_byte_comb = set(d_chunck_size_d_word_count[4].keys())
    set_word_5_byte_comb = set(d_chunck_size_d_word_count[5].keys())
    set_word_6_byte_comb = set(d_chunck_size_d_word_count[6].keys())

    print("len_set_word_1_byte_all: {}".format(len_set_word_1_byte_all))
    print("len_set_word_2_byte_all: {}".format(len_set_word_2_byte_all))
    print("len_set_word_3_byte_all: {}".format(len_set_word_3_byte_all))
    print("len_set_word_4_byte_all: {}".format(len_set_word_4_byte_all))
    print("len_set_word_5_byte_all: {}".format(len_set_word_5_byte_all))
    print("len_set_word_6_byte_all: {}".format(len_set_word_6_byte_all))

    print("len(set_word_2_byte_comb): {}".format(len(set_word_2_byte_comb)))
    print("len(set_word_3_byte_comb): {}".format(len(set_word_3_byte_comb)))
    print("len(set_word_4_byte_comb): {}".format(len(set_word_4_byte_comb)))
    print("len(set_word_5_byte_comb): {}".format(len(set_word_5_byte_comb)))
    print("len(set_word_6_byte_comb): {}".format(len(set_word_6_byte_comb)))

    def find_not_used_tpl_combinations(set_word_1_byte, set_word_n_byte, n=10):
        l_word_n_byte = list(set_word_n_byte)
        n_byte = len(l_word_n_byte[0])

        s_found_t = set()
        amount_try = 0

        for i in np.random.permutation(np.arange(0, len(l_word_n_byte))):
            l_orig = list(l_word_n_byte[i])

            for j in range(0, n_byte):
                l_cpy = list(l_orig)
                for t_1_byte in set_word_1_byte:
                    l_cpy[j] = t_1_byte[0]
                    t = tuple(l_cpy)
                    if not t in s_found_t and not t in set_word_n_byte:
                        s_found_t.add(t)
                        if n is not None and len(s_found_t) >= n:
                            return s_found_t
                    else:
                        amount_try += 1

            if n is not None and amount_try > 100000:
                return s_found_t

        return s_found_t

    set_word_bytes_to_change = set_word_2_byte_comb

    s_found_t_n_byte = find_not_used_tpl_combinations(set_word_1_byte_comb, set_word_bytes_to_change, n=None)
    assert len_set_word_2_byte_all == len(s_found_t_n_byte) + len(set_word_bytes_to_change)
    # assert len_set_word_2_byte_all == len(s_found_t_n_byte) + len(set_word_bytes_to_change)

    l_sorted_6_byte = sorted([(len(k)*v, len(k), v, k) for k, v in d_chunck_size_d_word_count[6].items()])
    l_sorted_7_byte = sorted([(len(k)*v, len(k), v, k) for k, v in d_chunck_size_d_word_count[7].items()])
    l_sorted_8_byte = sorted([(len(k)*v, len(k), v, k) for k, v in d_chunck_size_d_word_count[8].items()])

    d_word_pos = defaultdict(list)
    l_sorted_6_byte_some = l_sorted_6_byte[::-1][:10]
    for _, _, _, word in l_sorted_6_byte_some:
        print("word: {}".format(word))
        chunck_size = len(word)
        for i in range(0, length - chunck_size + 1):
            if i % 100000 == 0:
                print("chunck_size: {}, i: {}".format(chunck_size, i))
            if t_content[i:i+chunck_size] == word:
                d_word_pos[word].append(i)

    word = l_sorted_6_byte_some[1][3]
    print("word: {}".format(word))
    chunk_size_orig = len(word)
    chunck_size_new = 2

    arr_pos = np.array(d_word_pos[word])

    l_bytes_left = [t_content[i-chunck_size_new+1:i] for i in arr_pos]
    l_bytes_right = [t_content[i+chunk_size_orig:i+chunk_size_orig+chunck_size_new-1] for i in arr_pos]

    # check, if the left and right side is OK with the new found 
    found_new_word = False
    for word_new in s_found_t_n_byte:
        print("word_new: {}".format(word_new))
        found_new_word_inner = True
        for word_left in l_bytes_left:
            for i in range(0, chunck_size_new-1):
                t = word_left[i:] + word_new[:i+1]
                if t not in s_found_t_n_byte:
                    found_new_word_inner = False
                    break
            if found_new_word_inner == False:
                break

        if found_new_word_inner:
            for word_right in l_bytes_right:
                for i in range(0, chunck_size_new-1):
                    t = word_new[i+1:] + word_right[:i+1]
                    if t not in s_found_t_n_byte:
                        found_new_word_inner = False
                        break
                if found_new_word_inner == False:
                    break

        if found_new_word_inner:
            found_new_word = True
            break

    if found_new_word:
        print('Found word_new: "{}"'.format(word_new))
    else:
        print('Found word_new not found!!!')

    s_word_n_byte_new = set()

    for word_left in l_bytes_left:
        for i in range(0, chunck_size_new-1):
            t = word_left[i:] + word_new[:i+1]
            if t not in s_word_n_byte_new:
                s_word_n_byte_new.add(t)

    for word_right in l_bytes_right:
        for i in range(0, chunck_size_new-1):
            t = word_new[i+1:] + word_right[:i+1]
            if t not in s_word_n_byte_new:
                s_word_n_byte_new.add(t)

    print("len(s_word_n_byte_new): {}".format(len(s_word_n_byte_new)))

    # TODO: create a function with the word reductions!
    # TODO: create a function, where the word is checked, if is contains some symmetris or not!

    '''
    content_size = 1000000
    len(set_word_1_byte_all): 195
    len(set_word_2_byte_all): 38025
    len(set_word_3_byte_all): 7414875
    len(set_word_2_byte_comb): 5800
    len(set_word_3_byte_comb): 32465

    content_size = 10000000
    len(set_word_1_byte_all): 200
    len(set_word_2_byte_all): 40000
    len(set_word_3_byte_all): 8000000
    len(set_word_2_byte_comb): 11845
    len(set_word_3_byte_comb): 92713

    content_size = 100000000

    '''
    sys.exit()

    def combine_content_arr(t_content, rows):
        t_content = np.array(t_content)
        arr = np.array(t_content).reshape((1, -1))

        for i in range(1, rows):
            arr = np.vstack((arr[:, :-1], t_content[i:]))

        return arr

    arr_cont_rows = combine_content_arr(t_content[:10], rows=3)
    print("arr_cont_rows:\n{}".format(arr_cont_rows))

    # sys.exit()

    d = defaultdict(list)
    for chunck_size in range(2, 8):
        print("chunck_size: {}".format(chunck_size))
        # chunck_size = 3
        for i in range(0, len(t_content)-chunck_size+1):
            if i % 100000 == 0:
                print("i: {}".format(i))
            d[t_content[i:i+chunck_size]].append(i)

    print("chunck_size: {}".format(chunck_size))
    print("len(t_content): {}".format(len(t_content)))

    l_sorted = sorted([(len(v)*len(k), len(k), len(v), k) for k, v in d.items()])
    print('l_sorted[-100:]:')
    pprint(l_sorted[-100:])


