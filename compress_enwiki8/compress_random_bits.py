#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

from dotmap import DotMap

from operator import itemgetter
# from sortedcontainers import SortedSet

from collections import defaultdict

from os.path import expanduser
PATH_HOME = expanduser("~")+'/'
print("PATH_HOME: {}".format(PATH_HOME))

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
import global_object_getter_setter

import utils_compress_enwik8
# utils_compress_enwik8.do_some_simple_tests()
# sys.exit()

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

def create_dict_word_count_for_arr(arr, max_byte_length=10):
    d_arr_comb = {}
    d_arr_comb_unique = {}
    arr_comb = arr.copy().reshape((-1, 1))
    for comb_nr in range(1, max_byte_length+1):
        print("comb_nr: {}".format(comb_nr))
        d_arr_comb[comb_nr] = arr_comb
        arr_comb_view = arr_comb.reshape((-1, )).view(dtype=[('f{}'.format(i), '<u1') for i in range(0, comb_nr)])
        u, c = np.unique(arr_comb_view, return_counts=True)
        u = u.view('<u1').reshape((-1, arr_comb.shape[1]))
        idxs_sort = np.flip(np.argsort(c))
        u = u[idxs_sort]
        c = c[idxs_sort]

        # remove all occurences, which are only comming once!
        idxs_gt_1 = c>1
        u = u[idxs_gt_1]
        c = c[idxs_gt_1]

        d_arr_comb_unique[comb_nr] = {'u': u, 'c': c}
        
        arr_comb = np.hstack((arr_comb[:-1], arr[comb_nr:].reshape((-1, 1))))

    return d_arr_comb, d_arr_comb_unique


if __name__ == "__main__":
    OBJECT_NAME_ARR = 'arr_001'
    if not global_object_getter_setter.do_object_exist(OBJECT_NAME_ARR):
        arr = np.random.randint(0, 2, (100000, ), dtype=np.uint8)
        global_object_getter_setter.save_object(OBJECT_NAME_ARR, arr)
    else:
        arr = global_object_getter_setter.load_object(OBJECT_NAME_ARR)

    MAX_COMB_NR = 15
    OBJECT_NAME_D_ARR_COMB_DATA = 'd_'+OBJECT_NAME_ARR+'_comb_data_max_comb_nr_{}'.format(MAX_COMB_NR)
    if not global_object_getter_setter.do_object_exist(OBJECT_NAME_D_ARR_COMB_DATA):
        arr_comb = arr.reshape((-1, 1))
        arr1 = arr_comb.copy()

        d_arr_comb_data = {}
        for comb_nr in range(2, MAX_COMB_NR+1):
            print("comb_nr: {}".format(comb_nr))
            arr_comb = np.hstack((arr_comb[:-1], arr1[comb_nr-1:]))

            arr_comb_view = arr_comb.reshape((-1, )).view(dtype=[('f{}'.format(i), '<u1') for i in range(0, comb_nr)])
            u, c = np.unique(arr_comb_view, return_counts=True)
            u = u.view('<u1').reshape((-1, arr_comb.shape[1]))
            idxs_sort = np.flip(np.argsort(c))
            u = u[idxs_sort]
            c = c[idxs_sort]

            # remove all occurences, which are only comming once!
            idxs_gt_1 = c>1
            u = u[idxs_gt_1]
            c = c[idxs_gt_1]

            d_arr_comb_data[comb_nr] = {'arr_comb': arr_comb,'u': u, 'c': c}
        global_object_getter_setter.save_object(OBJECT_NAME_D_ARR_COMB_DATA, d_arr_comb_data)
    else:
        d_arr_comb_data = global_object_getter_setter.load_object(OBJECT_NAME_D_ARR_COMB_DATA)
    

    print("itemgetter('u', 'c')(d_arr_comb_data[4]): {}".format(itemgetter('u', 'c')(d_arr_comb_data[4])))



    sys.exit()

    arr = utils_compress_enwik8.get_arr(used_length=2**16)
    # arr = utils_compress_enwik8.get_arr(used_length=2**23)
    bytes_starting_size = arr.shape[0]
    # arr = utils_compress_enwik8.get_arr(used_length=2**22+1)
    
    OBJ_NAME_D_ARR_COMB = 'd_arr_comb_size_arr_{}'.format(arr.shape[0])
    OBJ_NAME_D_ARR_COMB_UNIQUE = 'd_arr_comb_unique_size_arr_{}'.format(arr.shape[0])

    # global_object_getter_setter.delete_object(OBJ_NAME_D_ARR_COMB)
    # global_object_getter_setter.delete_object(OBJ_NAME_D_ARR_COMB_UNIQUE)

    MAX_BYTE_LENGTH = 10

    l_content_bits_compressed = []
    l_arr_content_compressed_uint8 = []
    for round_nr in range(0, 15):
        if round_nr==0:
            if not global_object_getter_setter.do_object_exist(OBJ_NAME_D_ARR_COMB) or \
              not global_object_getter_setter.do_object_exist(OBJ_NAME_D_ARR_COMB_UNIQUE):
                d_arr_comb, d_arr_comb_unique = create_dict_word_count_for_arr(arr, max_byte_length=MAX_BYTE_LENGTH)

                global_object_getter_setter.save_object(OBJ_NAME_D_ARR_COMB, d_arr_comb)
                global_object_getter_setter.save_object(OBJ_NAME_D_ARR_COMB_UNIQUE, d_arr_comb_unique)
            else:
                d_arr_comb = global_object_getter_setter.load_object(OBJ_NAME_D_ARR_COMB)
                d_arr_comb_unique = global_object_getter_setter.load_object(OBJ_NAME_D_ARR_COMB_UNIQUE)
        else:
            d_arr_comb, d_arr_comb_unique = create_dict_word_count_for_arr(arr, max_byte_length=MAX_BYTE_LENGTH)

        l = []
        for byte_amount in range(2, MAX_BYTE_LENGTH+1):
            print("byte_amount: {}".format(byte_amount))
            arr_comb = d_arr_comb[byte_amount]
            for i_unique, unique_bytes in enumerate(d_arr_comb_unique[byte_amount]['u'][:20], 0):
                print("i_unique: {}".format(i_unique))
                idxs_nr = np.where(np.all(arr_comb==unique_bytes, axis=1))[0]
                idxs_ranges = np.vstack((idxs_nr, idxs_nr+byte_amount)).T
                l_idxs_ranges = utils_compress_enwik8.get_list_of_ranges(idxs_ranges)
                l.append(l_idxs_ranges[0])

                # is_somewhere_equal = False
                # unique_bytes_cpy = unique_bytes.copy()
                # for i in range(0, byte_amount-1):
                #     unique_bytes_cpy = np.roll(unique_bytes_cpy, 1)
                #     if np.all(unique_bytes_cpy==unique_bytes):
                #         is_somewhere_equal = True
                #         break

                # if is_somewhere_equal:
                #     print("somewhere equal for: i_unique: {}, unique_bytes: {}".format(i_unique, unique_bytes))
                #     break
            # break

        l_info = [((v.shape [0], v[0, 1]-v[0, 0]), v.shape[0]*(v[0, 1]-v[0, 0]), i, arr[v[0, 0]:v[0, 1]]) for i, v in enumerate(l, 0)]
        l_sorted = sorted(l_info, key=lambda x: x[1], reverse=True)
        
        l_chosen_index = [l_sorted[0][2]]
        l_chosen_unique_bytes = [l_sorted[0][3]]
        
        LEN_CHOSEN_INDEX = 16
        for _, _, i, unique_bytes in l_sorted:
            print("unique_bytes: {}".format(unique_bytes))
            # if any([utils_compress_enwik8.check_if_crossover_is_possible(unique_bytes, unique_bytes_2) for unique_bytes_2 in l_chosen_unique_bytes]):
            #     print("Ignore: i: {}, unique_bytes: {}".format(i, unique_bytes))
            #     continue
            
            is_crossover_possible = False
            for unique_bytes_2 in l_chosen_unique_bytes:
                if utils_compress_enwik8.check_if_crossover_is_possible(unique_bytes, unique_bytes_2):
                    is_crossover_possible = True
                    break
            if is_crossover_possible:
                print("Ignore: i: {}, unique_bytes: {}".format(i, unique_bytes))
                continue

            l_chosen_index.append(i)
            l_chosen_unique_bytes.append(unique_bytes)

            if len(l_chosen_index)==LEN_CHOSEN_INDEX:
            # if len(l_chosen_index)==8:
                break

        assert len(l_chosen_index)==LEN_CHOSEN_INDEX
        # l_chosen_index = [i for _, _, i, _ in l_sorted[:8]]
        l_chosen_idxs_ranges = [l[i] for i in l_chosen_index]
        idxs_ranges_combined = np.vstack(l_chosen_idxs_ranges) 
        idxs_ranges_sorted = idxs_ranges_combined[np.argsort(idxs_ranges_combined[:, 0])]
        assert np.all(idxs_ranges_sorted[1:, 0]>=idxs_ranges_sorted[:-1, 1])
        
        arr_diff = np.hstack(((idxs_ranges_sorted[0, 0], ), idxs_ranges_sorted[1:, 0]-idxs_ranges_sorted[:-1, 1]))
        arr_diff_bits_amount = [len(bin(v)[2:]) if v>0 else 0 for v in arr_diff]

        # first calculate, how many bytes there really are!
        l_amounts = [l_info[i][1] for i in l_chosen_index]
        bytes_amount_raw = sum(l_amounts)
        bits_amount_raw = bytes_amount_raw*8

        # generate a mapping index of the unique bytes
        l_chosen_ub_tbl = [tuple(v.tolist()) for v in l_chosen_unique_bytes]

        d_chosen_ub_tbl = {t: i for i, t in enumerate(l_chosen_ub_tbl, 0)}
        l_idx_tbl = [d_chosen_ub_tbl[tuple(arr[i1:i2].tolist())] for i1, i2 in idxs_ranges_sorted]

        length = len(l_idx_tbl)
        length_needed_bytes = (lambda x: (x+x%2)//2)(len(hex(length)[2:]))

        content_bits_compressed = (
            bin(len(l_chosen_ub_tbl)-1)[2:].zfill(4)+
            ''.join([bin(len(v)-1)[2:].zfill(4) for v in l_chosen_unique_bytes])+
            ''.join([bin(v)[2:].zfill(8) for v in np.hstack(l_chosen_unique_bytes)])+
            bin(length_needed_bytes-1)[2:].zfill(2)+
            bin(length)[2:].zfill(8*length_needed_bytes)+
            ''.join([bin(i)[2:].zfill(4) for i in l_idx_tbl])+
            ''.join([bin(i)[2:].zfill(4) for i in arr_diff_bits_amount])+
            ''.join([bin(v)[2:] for v in arr_diff if v>0])
        )
        print("len(content_bits_compressed): {}".format(len(content_bits_compressed)))
        print("bits_amount_raw: {}".format(bits_amount_raw))

        if len(content_bits_compressed)%8!=0:
            length_content = len(content_bits_compressed)
            content_bits_compressed += '0'*(8-(length_content%8))

        arr_content_compressed = np.sum(np.array(list(map(int, list(content_bits_compressed)))).reshape((-1, 8))*2**np.arange(7, -1, -1), axis=1)
        arr_content_compressed_uint8 = arr_content_compressed.astype(np.uint8)

        arr_content_compressed_uint8.tofile('content_compressed_round_nr_{}.hex'.format(round_nr))

        idxs_rest = ~np.isin(np.arange(0, len(arr)), np.hstack([np.arange(i1, i2) for i1, i2 in idxs_ranges_sorted]))
        arr = arr[idxs_rest]

        l_content_bits_compressed.append(content_bits_compressed)
        l_arr_content_compressed_uint8.append(arr_content_compressed_uint8)

        print("Rest: len(arr): {}".format(len(arr)))

    np.hstack(l_arr_content_compressed_uint8).tofile('content_compressed_full.hex')

    bytes_compressed_size = sum([a.shape[0] for a in l_arr_content_compressed_uint8])+arr.shape[0]
    print("bytes_starting_size: {}".format(bytes_starting_size))
    print("bytes_compressed_size: {}".format(bytes_compressed_size))
