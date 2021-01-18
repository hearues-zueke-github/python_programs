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

# m byte word ... longer word
# n byte word ... shorter word
def find_best_new_word_substitute(t_content, d_chunck_size_d_word_count, word_length_old, word_length_new):
    # word_length_new = 2
    # word_length_old = 6
    set_word_bytes_new = set(d_chunck_size_d_word_count[word_length_new].keys())

    set_word_1_byte_comb = set(d_chunck_size_d_word_count[1].keys())
    s_found_t_n_byte_not_used = find_not_used_tpl_combinations(set_word_1_byte_comb, set_word_bytes_new, n=None)
    assert d_len_set_word_all[word_length_new] == len(s_found_t_n_byte_not_used) + len(set_word_bytes_new)

    d_word_count_old = d_chunck_size_d_word_count[word_length_old]
    l_sorted_m_byte = sorted([(len(k)*v, len(k), v, k) for k, v in d_word_count_old.items()])

    length_content = len(t_content)
    l_word_m_choosen = []
    d_word_m_pos = defaultdict(list)
    l_sorted_m_byte_some = l_sorted_m_byte[::-1][:10]
    for _, _, _, word in l_sorted_m_byte_some:
        chunck_size = len(word)
        print("chunck_size: {}, word: {}".format(chunck_size, word))

        arr_words = np.tile(word, word_length_old-1).reshape((word_length_old-1, word_length_old))
        for i in range(0, word_length_old - 1):
            arr_words[i] = np.roll(arr_words[i], i + 1)
        if np.any(np.all(arr_words == word, axis=1)):
            continue

        l_word_m_choosen.append(word)

        for i in range(0, length_content - chunck_size + 1):
            # if i % 100000 == 0:
            #     print("chunck_size: {}, i: {}".format(chunck_size, i))
            if t_content[i:i+chunck_size] == word:
                d_word_m_pos[word].append(i)

    print('l_word_m_choosen')
    pprint(l_word_m_choosen)

    word_old = l_word_m_choosen[0]
    print("word_old: {}".format(word_old))
    word_length_old = len(word_old)
    word_length_new = 2

    arr_pos = np.array(d_word_m_pos[word_old])
    assert arr_pos[0] >= 0
    assert np.all(np.diff(arr_pos) >= word_length_old)
    assert arr_pos[-1] <= len(t_content) - word_length_old + 1

    l_bytes_left = [t_content[i-word_length_new + 1:i] for i in arr_pos]
    l_bytes_right = [t_content[i+word_length_old:i+word_length_old+word_length_new-1] for i in arr_pos]

    # check, if the left and right side is OK with the new found 
    found_new_word = False
    l_found_t_n_byte_not_used = list(s_found_t_n_byte_not_used)
    for word_new in [l_found_t_n_byte_not_used[i] for i in np.random.permutation(np.arange(0, len(l_found_t_n_byte_not_used)))]:
        # print("word_new: {}".format(word_new))
        found_new_word_inner = True
        for word_left in l_bytes_left:
            for i in range(0, word_length_new - 1):
                t = word_left[i:] + word_new[:i+1]
                if t not in s_found_t_n_byte_not_used:
                    found_new_word_inner = False
                    break
            if found_new_word_inner == False:
                break

        if found_new_word_inner:
            for word_right in l_bytes_right:
                for i in range(0, word_length_new-1):
                    t = word_new[i+1:] + word_right[:i+1]
                    if t not in s_found_t_n_byte_not_used:
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
        print('No word_new found!!!')
        # sys.exit()

    if not found_new_word:
        return None, None, None

    return word_old, word_new, arr_pos


def get_new_reduced_content(t_content, word_old, word_length_old, word_new, word_length_new, arr_pos):
# get the new t_content variable!
    l_content_new = []
    if arr_pos[0] != 0:
        l_content_new.extend(t_content[:arr_pos[0]])

    if len(arr_pos) == 1:
        l_content_new.extend(word_new)
    elif len(arr_pos) > 1:
        for pos1, pos2 in zip(arr_pos[:-1], arr_pos[1:]):
            l_content_new.extend(word_new)
            l_content_new.extend(t_content[pos1+word_length_old:pos2])

        l_content_new.extend(word_new)

    if arr_pos[-1] != len(t_content) - word_length_old + 1:
        l_content_new.extend(t_content[arr_pos[-1]+word_length_old:])

    return tuple(l_content_new)


if __name__ == '__main__':
    with open("/home/doublepmcl/Downloads/enwik8", 'rb') as f:
        content = f.read()

    # content_size = 100000
    content_size = 1000000
    # content_size = 10000000
    # content_size = 100000000

    t_content = tuple(map(int, content[:content_size]))

    d_chunck_size_d_word_count = defaultdict(lambda: defaultdict(int))

    folder_path_enwik8_files = TEMP_DIR + 'enwik8_files/'
    mkdirs(folder_path_enwik8_files)
    folder_path = TEMP_DIR + 'd_word_count/'
    mkdirs(folder_path)
    max_chunck_size = 8
    
    length_orig = len(t_content)
    for chunck_size in range(1, max_chunck_size+1):
        file_name = 'chunck_size_{}_length_{}.pkl.gz'.format(chunck_size, length_orig)
        obj_path = folder_path + file_name
        if not os.path.exists(obj_path):
            print("Creating object: obj_path: {}".format(obj_path))
            d = d_chunck_size_d_word_count[chunck_size]
            for i in range(0, length_orig - chunck_size + 1):
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

    len_set_word_1_byte_all = len(set(d_chunck_size_d_word_count[1].keys()))
    d_len_set_word_all = {i: len_set_word_1_byte_all**i for i in range(1, max_chunck_size+1)}

    l_old_new_word = []

    with open(folder_path_enwik8_files + 'enwik8_size_{}.txt'.format(len(t_content)), 'wb') as f:
        f.write(bytes(t_content))

    for rounds in range(0, 50):
        print("rounds: {}".format(rounds))

        word_length_old = 5
        word_length_new = 2
        word_old, word_new, arr_pos = find_best_new_word_substitute(
            t_content=t_content,
            d_chunck_size_d_word_count=d_chunck_size_d_word_count,
            word_length_old=word_length_old,
            word_length_new=word_length_new,
        )

        l_old_new_word.append((word_old, word_new))

        t_content_new = get_new_reduced_content(
            t_content=t_content,
            word_old=word_old,
            word_length_old=word_length_old,
            word_new=word_new,
            word_length_new=word_length_new,
            arr_pos=arr_pos,
        )
        print("len(t_content_new): {}".format(len(t_content_new)))

        d_chunck_size_d_word_count_new = defaultdict(lambda: defaultdict(int))

        length_new = len(t_content_new)
        for chunck_size in range(1, max_chunck_size+1):
            print("rounds: {}, chunck_size: {}".format(rounds, chunck_size))
            d = d_chunck_size_d_word_count_new[chunck_size]
            for i in range(0, length_new - chunck_size + 1):
                # if i % 100000 == 0:
                #     print("chunck_size: {}, i: {}".format(chunck_size, i))
                d[t_content_new[i:i+chunck_size]] += 1

        t_content = t_content_new
        d_chunck_size_d_word_count = d_chunck_size_d_word_count_new

        for chunck_size in range(1, max_chunck_size+1):
            print("len(d_chunck_size_d_word_count[{}]): {}".format(chunck_size, len(d_chunck_size_d_word_count[chunck_size])))

        print('-'*40)
        print("len(t_content): {}".format(len(t_content)))
        print('-'*40)

    content_full = bytes(tuple(itertools.chain(*[tuple(itertools.chain(*t)) for t in l_old_new_word])) + t_content)
    print("Final (not finished yet! in Bytes): len(content_full): {}".format(len(content_full)))
    with open(folder_path_enwik8_files + 'enwik8_size_{}_rounds_{}_new_size_{}.txt'.format(content_size, rounds, len(content_full)), 'wb') as f:
        f.write(content_full)
        
    # break
