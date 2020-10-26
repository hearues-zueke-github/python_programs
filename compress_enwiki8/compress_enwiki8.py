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

import utils_compress_enwik8

def create_dict_word_count_for_arr(arr, max_byte_length=10):
    d_arr_comb = {}
    d_arr_comb_unique = {}
    arr_comb = arr.copy().reshape((-1, 1))
    for comb_nr in range(1, max_byte_length+1):
        print("creating comb_nr: {}".format(comb_nr))
        d_arr_comb[comb_nr] = arr_comb
        arr_comb_view = arr_comb.reshape((-1, )).view(dtype=[('f{}'.format(i), '<u1') for i in range(0, 2)])
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

    # arr = utils_compress_enwik8.get_arr(used_length=2**21)
    # arr = utils_compress_enwik8.get_arr(used_length=2**18)
    arr = utils_compress_enwik8.get_arr(used_length=-1)
    # arr = utils_compress_enwik8.get_arr(used_length=2**23)
    # arr = utils_compress_enwik8.get_arr(used_length=2**22+1)


    used_len = 1000000
    next_step = used_len // 4
    max_len = 10
    d_sum = {i: {} for i in range(2, max_len+1)}
    d_stats = {i: [] for i in range(2, max_len+1)}
    for pos_i in range(0, used_len*10 - used_len + 1, next_step):
    # for pos_i in range(0, arr.shape[0], next_step):
        arr_1 = arr[pos_i:pos_i+used_len+max_len].reshape((-1, 1))
        print("pos_i: {:9}, {:9}".format(pos_i, pos_i+used_len))
        for i in range(2, max_len+1):
            arr_1 = np.hstack((arr_1[:-1], arr_1[1:, -1:]))
            u, c = np.unique(arr_1.reshape((-1, )).view(','.join(['u1']*i)), return_counts=True)
            d_1 = {tuple(t): j for t, j in zip(u, c)}

            d = d_sum[i]

            for t, j in d_1.items():
                if t not in d:
                    d[t] = j
                else:
                    d[t] += j

            # get the max len for each seperate combined bytes!
            l_t, l_j = list(zip(*list(d.items())))
            i_max = np.argmax(l_j)

            print("- i: {:2}, amount: {:10}, mult: {:10}, t: {}".format(i, l_j[i_max], i*l_j[i_max], l_t[i_max]))

        for i in range(2, max_len+1):
            d = d_sum[i]
            l = list(d.items())
            l_sort = sorted(list(d.items()), reverse=True, key=lambda x: (x[1], x[0]))

            d_stat = d_stats[i]

            d_stat.append('{:9},{:9}:{}'.format(
                pos_i,
                pos_i+used_len,
                '|'.join(['{},{:5}'.format(''.join(map(lambda x: '{:02X}'.format(x), t)), c) for t, c in l_sort[:5]])
            ))

    with open(TEMP_DIR+'compress_enwik8_stats.txt', 'w') as f:
        for i in range(2, max_len+1):
            l = d_stats[i]

            f.write('len: {:2}\n'.format(i))
            f.write('\n'.join(l)+'\n'+'\n')

    sys.exit()


    bytes_starting_size = arr.shape[0]

    # global_object_getter_setter.delete_object(OBJ_NAME_D_ARR_COMB)
    # global_object_getter_setter.delete_object(OBJ_NAME_D_ARR_COMB_UNIQUE)


    # LEN_CHOSEN_INDEX = 2**5
    # LEN_BITS_CHOSEN_INDEX = len(bin(LEN_CHOSEN_INDEX-1))-2

    argv = sys.argv
    LEN_BITS_CHOSEN_INDEX = int(argv[1])
    # LEN_BITS_CHOSEN_INDEX = 2
    LEN_CHOSEN_INDEX = 2**LEN_BITS_CHOSEN_INDEX

    MAX_BYTE_LENGTH = 8
    MAX_FIRST_POS = 10
    OBJ_NAME_D_ARR_COMB = 'd_arr_comb_size_arr_{}_max_byte_length_{}'.format(arr.shape[0], MAX_BYTE_LENGTH)
    OBJ_NAME_D_ARR_COMB_UNIQUE = 'd_arr_comb_unique_size_arr_{}_max_byte_length_{}'.format(arr.shape[0], MAX_BYTE_LENGTH)

    ROUNDS_AMOUNT = 5

    l_content_bits_compressed = []
    l_arr_content_compressed_uint8 = []
    l_len_raw_bits = []
    l_len_compressed_bits = []
    for round_nr in range(0, ROUNDS_AMOUNT):
        print("round_nr: {}".format(round_nr))
        # round_nr = 0
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

        if all([d_arr_comb_unique[i]['u'].shape[0]==0 for i in range(2, MAX_BYTE_LENGTH+1)]):
            print("All arr_comb unique empty!")
            break

        d_word_occurences = {}
        d_word_weight = {}
        for comb_nr in range(2, MAX_BYTE_LENGTH+1):
            print("comb_nr: {}".format(comb_nr))
            arr_comb = d_arr_comb[comb_nr]
            arr_comb_unique = d_arr_comb_unique[comb_nr]
            u = arr_comb_unique['u']
            c = arr_comb_unique['c']

            for unique_bytes, amount in zip(u, c):
                t = tuple(unique_bytes.tolist())
                d_word_occurences[t] = amount
                # d_word_weight[t] = amount*2**len(t)
                # d_word_weight[t] = amount*2**(len(bin(len(t))[2:])-1)
                # d_word_weight[t] = amount*2**len(bin(len(t))[2:])
                d_word_weight[t] = amount*len(t)
                # d_word_weight[t] = amount*len(bin(len(t))[2:])

        d_saved_compressions = {}
        for first_pos in range(0, MAX_FIRST_POS):
            print("first_pos: {}".format(first_pos))

            unique_bytes_index = 0
            l_chosen_unique_bytes_index = []
            l_chosen_unique_bytes_diff = []
            l_chosen_unique_bytes_ranges = []
            d_chosen_unique_bytes_to_index = {}
            length = arr.shape[0]
            pos = first_pos
            last_pos = 0
            while pos < length:
                l_tpls = []
                l_weights = []
                for j in range(2, MAX_BYTE_LENGTH):
                    t = tuple(arr[pos:pos+j].tolist())
                    if t in d_word_weight:
                        l_weights.append(d_word_weight[t])
                        l_tpls.append(t)

                if len(l_weights)==0:
                    pos += 1
                    continue

                t = l_tpls[np.argmax(l_weights)]

                if not t in d_chosen_unique_bytes_to_index:
                    if unique_bytes_index<LEN_CHOSEN_INDEX:
                        d_chosen_unique_bytes_to_index[t] = unique_bytes_index
                        unique_bytes_index += 1
                    else:
                        pos += 1
                        continue

                len_t = len(t)
                l_chosen_unique_bytes_index.append(d_chosen_unique_bytes_to_index[t])
                l_chosen_unique_bytes_diff.append(pos-last_pos)
                l_chosen_unique_bytes_ranges.append((pos, pos+len_t))
                last_pos = pos+len_t
                pos = last_pos

            # assert unique_bytes_index==LEN_CHOSEN_INDEX
            if unique_bytes_index!=LEN_CHOSEN_INDEX:
                continue

            # # assert check
            # pos = 0
            # l_chosen_unique_bytes = [v for _, v in sorted([(v, k) for k, v in d_chosen_unique_bytes_to_index.items()])]
            # for i in range(0, len(l_chosen_unique_bytes_index)):
            #     ub_index = l_chosen_unique_bytes_index[i]
            #     ub_tpl = l_chosen_unique_bytes[ub_index]

            #     ub_diff = l_chosen_unique_bytes_diff[i]
            #     pos += ub_diff
            #     length = len(ub_tpl)

            #     ub_arr = arr[pos:pos+length]
            #     assert tuple(ub_arr.tolist())==ub_tpl
            #     pos += len(ub_tpl)

            l_diff_bits_amount = [len(bin(v)[2:]) if v>0 else 0 for v in l_chosen_unique_bytes_diff]

            # TODO 2020.04.13: find a better way for representing the compressed data
            u_index_orig, c_index_orig = np.unique(l_chosen_unique_bytes_index, return_counts=True)
            idxs_index = [v[2] for v in sorted(list(zip(c_index_orig, u_index_orig, range(0, len(u_index_orig)))), reverse=True)]
            u_index = u_index_orig[idxs_index]
            c_index = c_index_orig[idxs_index]

            u_diff_orig, c_diff_orig = np.unique(l_chosen_unique_bytes_diff, return_counts=True)
            idxs_diff = [v[2] for v in sorted(list(zip(c_diff_orig, u_diff_orig, range(0, len(u_diff_orig)))), reverse=True)]
            u_diff = u_diff_orig[idxs_diff]
            c_diff = c_diff_orig[idxs_diff]

            assert np.sum(c_index)==np.sum(c_diff)

            # l_chosen_unique_bytes = [word for _, word in sorted([(v, k) for k, v in d_chosen_unique_bytes_to_index.items()])]
            # l_chosen_unique_bytes_cpy = deepcopy(l_chosen_unique_bytes)
            # print("before: l_chosen_unique_bytes: {}".format(l_chosen_unique_bytes))

            # change all content of index and diff to the new sorted by occurences one!
            # d_index = {i_new: i_prev for i_prev, i_new in zip(u_index_orig, u_index)}
            # d_index = {i_prev: i_new for i_prev, i_new in zip(u_index_orig, u_index)}
            d_index = dict([(v, i) for i, v in enumerate(idxs_index, 0)])

            # l_chosen_unique_bytes = [l_chosen_unique_bytes[i] for i in np.argsort(u_index)]
            # l_chosen_unique_bytes = [l_chosen_unique_bytes[d_index[i]] for i in np.argsort(u_index)]
            # l_chosen_unique_bytes = [l_chosen_unique_bytes[d_index[i]] for i in range(0, len(u_index))]
            # print("after: l_chosen_unique_bytes: {}".format(l_chosen_unique_bytes))

            # print("before: l_chosen_unique_bytes_index: {}".format(l_chosen_unique_bytes_index[:100]))
            # l_chosen_unique_bytes_index_cpy = deepcopy(l_chosen_unique_bytes_index)
            # l_chosen_unique_bytes_index = [idxs_index[i] for i in l_chosen_unique_bytes_index]
            l_chosen_unique_bytes_index = [d_index[i] for i in l_chosen_unique_bytes_index]
            # print("after: l_chosen_unique_bytes_index: {}".format(l_chosen_unique_bytes_index[:100]))
            d_chosen_unique_bytes_to_index = {k: d_index[v] for k, v in d_chosen_unique_bytes_to_index.items()}
            l_chosen_unique_bytes = [word for _, word in sorted([(v, k) for k, v in d_chosen_unique_bytes_to_index.items()])]

            u, c = np.unique(l_chosen_unique_bytes_index, return_counts=True)
            assert np.all(u_index_orig==u)
            assert np.all(c_index==c)

            d_diff_to_idx = {k: i for i, k in enumerate(u_diff, 0)}
            l_chosen_unique_bytes_diff_tbl = u_diff.tolist()
            l_chosen_unique_bytes_diff_idx = [d_diff_to_idx[i] for i in l_chosen_unique_bytes_diff]

            length = len(l_chosen_unique_bytes_index)
            length_needed_bytes = (lambda x: (x+x%2)//2)(len(hex(length)[2:]))

            length_diff_tbl = len(l_chosen_unique_bytes_diff_tbl)
            length_diff_tbl_needed_bytes = (lambda x: (x+x%2)//2)(len(hex(length_diff_tbl)[2:]))

            assert len(l_chosen_unique_bytes_index)==len(l_chosen_unique_bytes_diff_idx)

            # assert check
            pos = 0
            # l_chosen_unique_bytes = [v for _, v in sorted([(v, k) for k, v in d_chosen_unique_bytes_to_index.items()])]
            for i in range(0, len(l_chosen_unique_bytes_index)):
                ub_index = l_chosen_unique_bytes_index[i]
                ub_tpl = l_chosen_unique_bytes[ub_index]

                ub_diff = l_chosen_unique_bytes_diff[i]
                pos += ub_diff
                length = len(ub_tpl)

                # print("ub_index: {}, ub_tpl: {}".format(ub_index, ub_tpl))
                # print("ub_diff: {}, pos: {}".format(ub_diff, pos))
                # print("length: {}".format(length))

                ub_arr = arr[pos:pos+length]
                # print("ub_arr: {}, ub_tpl: {}".format(ub_arr, ub_tpl))
                assert tuple(ub_arr.tolist())==ub_tpl
                pos += len(ub_tpl)

            needed_bits_idx_index = np.sum(c_index*[k for j in [(2*i, )*2**i for i in range(1, 12)] for k in j][:u_index.shape[0]])
            needed_bits_idx_diff = np.sum(c_diff*[k for j in [(2*i, )*2**i for i in range(1, 12)] for k in j][:u_diff.shape[0]])
            # print("needed_bits_idx_index: {}".format(needed_bits_idx_index))
            # print("needed_bits_idx_diff: {}".format(needed_bits_idx_diff))

            needed_bits_word_tbl = 8+4*len(u_index)+8*sum(map(len, l_chosen_unique_bytes))
            needed_bits_diff_tbl = 2+length_diff_tbl_needed_bytes*8+4*len(u_diff)+sum(map(lambda x: len(bin(x)[2:]), l_chosen_unique_bytes_diff_tbl))
            needed_bits_idx_word_diff = 2+length_needed_bytes*8+needed_bits_idx_index+needed_bits_idx_diff

            # print("- needed_bits_word_tbl: {}".format(needed_bits_word_tbl))
            # print("- needed_bits_diff_tbl: {}".format(needed_bits_diff_tbl))
            # print("- needed_bits_idx_word_diff: {}".format(needed_bits_idx_word_diff))
            print("- needed bits: (word_tbl, diff_tbl, idx_word_diff): {}".format((needed_bits_word_tbl, needed_bits_diff_tbl, needed_bits_idx_word_diff)))

            # # old variant
            # content_bits_compressed_str = (
            #     bin(MAX_BYTE_LENGTH)[2:].zfill(8)+
            #     ''.join([bin(len(v)-1)[2:].zfill(4) for v in l_chosen_unique_bytes])+
            #     ''.join([bin(v)[2:].zfill(8) for v in np.hstack(l_chosen_unique_bytes)])+
            #     bin(length_needed_bytes-1)[2:].zfill(2)+
            #     bin(length)[2:].zfill(8*length_needed_bytes)+
            #     ''.join([bin(i)[2:].zfill(LEN_BITS_CHOSEN_INDEX) for i in l_chosen_unique_bytes_index])+
            #     ''.join([bin(i)[2:].zfill(4) for i in l_diff_bits_amount])+
            #     ''.join([bin(v)[2:] for v in l_chosen_unique_bytes_diff if v>0])
            # )


            # first, create the corresponding str_bits for the number in a list!
            max_len = max([u_index.shape[0], u_diff.shape[0]])
            l_bits_coding_basic = ['00', '01']
            l_bits_coding_accumulative = deepcopy(l_bits_coding_basic)
            l_bits_coding = deepcopy(l_bits_coding_basic)
            l_bits_coding_extend = ['10', '11']
            while len(l_bits_coding)<max_len:
                l_bits_coding_accumulative = [b1+b2 for b1 in l_bits_coding_extend for b2 in l_bits_coding_accumulative]
                l_bits_coding += l_bits_coding_accumulative
            l_bits_coding = l_bits_coding[:max_len]
            # sys.exit()

            # new variant
            content_bits_compressed_str_word_tbl = (
                bin(MAX_BYTE_LENGTH)[2:].zfill(8)+
                ''.join([bin(len(v)-1)[2:].zfill(4) for v in l_chosen_unique_bytes])+
                ''.join([bin(v)[2:].zfill(8) for v in np.hstack(l_chosen_unique_bytes)])
            )
            content_bits_compressed_str_diff_tbl = (
                bin(length_diff_tbl_needed_bytes-1)[2:].zfill(2)+
                bin(length_diff_tbl)[2:].zfill(8*length_diff_tbl_needed_bytes)+
                ''.join([bin(len(bin(v)[2:])-1)[2:].zfill(4) for v in l_chosen_unique_bytes_diff_tbl])+
                ''.join([bin(v)[2:] for v in l_chosen_unique_bytes_diff_tbl])
            )
            content_bits_compressed_str_idx_word_diff = (
                bin(length_needed_bytes-1)[2:].zfill(2)+
                bin(length)[2:].zfill(8*length_needed_bytes)+
                ''.join([l_bits_coding[i_index]+l_bits_coding[i_diff_idx] for i_index, i_diff_idx in zip(l_chosen_unique_bytes_index, l_chosen_unique_bytes_diff_idx)])
            )

            assert needed_bits_word_tbl==len(content_bits_compressed_str_word_tbl)
            assert needed_bits_diff_tbl==len(content_bits_compressed_str_diff_tbl)
            assert needed_bits_idx_word_diff==len(content_bits_compressed_str_idx_word_diff)

            content_bits_compressed_str = (
                content_bits_compressed_str_word_tbl+
                content_bits_compressed_str_diff_tbl+
                content_bits_compressed_str_idx_word_diff
            )



            len_bytes_raw = sum([len(l_chosen_unique_bytes[i]) for i in l_chosen_unique_bytes_index])
            len_bits_raw = len_bytes_raw*8

            # print("l_chosen_unique_bytes: {}".format(l_chosen_unique_bytes))
            l_lens = list(map(len, l_chosen_unique_bytes))
            # print("l_lens: {}".format(l_lens))


            if len(content_bits_compressed_str)%8!=0:
                length_content = len(content_bits_compressed_str)
                content_bits_compressed_str += '0'*(8-(length_content%8))
            len_bits_compressed = len(content_bits_compressed_str)

            print("- bits: raw: {} compressed: {}, percent: {:.06f}%".format(len_bits_raw, len_bits_compressed, (len_bits_raw-len_bits_compressed)/len_bits_raw*100))

            arr_content_compressed = np.sum(np.array(list(map(int, list(content_bits_compressed_str)))).reshape((-1, 8))*2**np.arange(7, -1, -1), axis=1)
            arr_content_compressed_uint8 = arr_content_compressed.astype(np.uint8)

            d_saved_compressions[first_pos] = dict(
                l_chosen_unique_bytes_index=l_chosen_unique_bytes_index,
                l_chosen_unique_bytes_diff=l_chosen_unique_bytes_diff,
                l_chosen_unique_bytes_diff_tbl=l_chosen_unique_bytes_diff_tbl,
                l_chosen_unique_bytes_diff_idx=l_chosen_unique_bytes_diff_idx,
                l_chosen_unique_bytes_ranges=l_chosen_unique_bytes_ranges,
                d_chosen_unique_bytes_to_index=d_chosen_unique_bytes_to_index,
                content_bits_compressed_str=content_bits_compressed_str,
                arr_content_compressed_uint8=arr_content_compressed_uint8,
                len_bits_raw=len_bits_raw,
                len_bits_compressed=len_bits_compressed,
                l_lens=l_lens,
                percent_compression=(len_bits_raw-len_bits_compressed)/len_bits_raw,
                max_diff=max(l_chosen_unique_bytes_diff),
                max_diff_bits=len(bin(max(l_chosen_unique_bytes_diff))[2:]),
            )

        if len(d_saved_compressions)==0:
            print("No more valid d_saved_compressions found!")
            break

        l_lens_table = [(k2, [len(v) for _, v in sorted([(v, k) for k, v in d_saved_compressions[k2]['d_chosen_unique_bytes_to_index'].items()])]) for k2 in sorted(d_saved_compressions.keys())]
        # print("l_lens_table: {}".format(l_lens_table))
        print("l_lens_table:")
        for k, v in l_lens_table:
            print("k: {}, v: {}".format(k, v))
        l_percent_compressed = [(k2, d_saved_compressions[k2]['percent_compression'], '{:.06f}%'.format(d_saved_compressions[k2]['percent_compression']*100)) for k2 in sorted(d_saved_compressions.keys())]
        # TODO 2020.04.12: add a simple check, where the best percentage is used!
        print("l_percent_compressed: {}".format(l_percent_compressed))
        best_pos_start = sorted(l_percent_compressed, key=lambda x: (x[1], ), reverse=True)[0][0]
        print("best_pos_start: {}".format(best_pos_start))
        print("- l_percent_compressed[best_pos_start]: {}".format(l_percent_compressed[best_pos_start]))

        # sys.exit()
        d = d_saved_compressions[best_pos_start]
        len_bits_raw = d['len_bits_raw']
        len_bits_compressed = d['len_bits_compressed']
        # l_chosen_unique_bytes_index = d['l_chosen_unique_bytes_index']
        # l_chosen_unique_bytes_diff = d['l_chosen_unique_bytes_diff']
        l_chosen_unique_bytes_ranges = d['l_chosen_unique_bytes_ranges']
        content_bits_compressed_str = d['content_bits_compressed_str']
        arr_content_compressed_uint8 = d['arr_content_compressed_uint8']

        if LEN_CHOSEN_INDEX!=len(l_chosen_unique_bytes):
            break
        # assert LEN_CHOSEN_INDEX==len(l_chosen_unique_bytes)

        l_len_raw_bits.append(len_bits_raw)
        l_len_compressed_bits.append(len_bits_compressed)

        # arr_content_compressed_uint8.tofile('content_compressed_round_nr_{}.hex'.format(round_nr))

        idxs_rest = ~np.isin(np.arange(0, len(arr)), np.hstack([np.arange(i1, i2) for i1, i2 in l_chosen_unique_bytes_ranges]))
        # idxs_rest = ~np.isin(np.arange(0, len(arr)), np.hstack([np.arange(i1, i2) for i1, i2 in idxs_ranges_sorted]))
        arr = arr[idxs_rest]

        l_content_bits_compressed.append(content_bits_compressed_str)
        l_arr_content_compressed_uint8.append(arr_content_compressed_uint8)

        # sys.exit()

    print()
    print("LEN_BITS_CHOSEN_INDEX: {}".format(LEN_BITS_CHOSEN_INDEX))
    print("LEN_CHOSEN_INDEX: {}".format(LEN_CHOSEN_INDEX))
    print("MAX_BYTE_LENGTH: {}".format(MAX_BYTE_LENGTH))
    print("ROUNDS_AMOUNT: {}".format(ROUNDS_AMOUNT))

    print("l_len_raw_bits: {}".format(l_len_raw_bits))
    print("l_len_compressed_bits: {}".format(l_len_compressed_bits))

    sum_raw_bits = sum(l_len_raw_bits)
    sum_compressed_bits = sum(l_len_compressed_bits)
    percent_compression = ((sum_raw_bits-sum_compressed_bits)/sum_raw_bits)*100

    print("sum_raw_bits: {}".format(sum_raw_bits))
    print("sum_compressed_bits: {}".format(sum_compressed_bits))
    missing_raw_bits = bytes_starting_size*8-sum_raw_bits
    print("missing_raw_bits: {}".format(missing_raw_bits))
    print("percent_compression (without rest raw): {:.06f} %".format(percent_compression))


    # save first the rest raw bytes then afterwards the compressed one!
    arr_len = np.array((arr.shape[0], ), dtype=np.uint32)
    arr_compressed_full = np.hstack([arr_len.view(dtype=np.uint8), arr]+l_arr_content_compressed_uint8)

    TMP_PATH_DIR = tempfile.gettempdir()
    tmp_hex_dir = os.path.join(TMP_PATH_DIR, 'compressed_files_enwik8/')
    if not os.path.exists(tmp_hex_dir):
        os.makedirs(tmp_hex_dir)

    print("bytes_starting_size: {}".format(bytes_starting_size))

    arr_compressed_full.tofile(
        (tmp_hex_dir+'content_compressed_size_orig_{size_orig}_size_comp_{size_comp}_round_nr_{round_nr}'+
        '_max_idx_{LEN_CHOSEN_INDEX}_max_word_len_{MAX_BYTE_LENGTH}_max_first_pos_{MAX_FIRST_POS}.hzzv2.hex').format(
            size_orig=bytes_starting_size,
            size_comp=arr_compressed_full.shape[0],
            round_nr=round_nr,
            LEN_CHOSEN_INDEX=LEN_CHOSEN_INDEX,
            MAX_BYTE_LENGTH=MAX_BYTE_LENGTH,
            MAX_FIRST_POS=MAX_FIRST_POS,
        )
    )

    perc_comp_with_rest_raw = (bytes_starting_size-arr_compressed_full.shape[0])/bytes_starting_size*100
    print("perc_comp_with_rest_raw: {:.06f} %".format(perc_comp_with_rest_raw))

    sys.exit()



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

    l_info = [((v.shape [0], v[0, 1]-v[0, 0]), v.shape[0]*(v[0, 1]-v[0, 0]), i, arr[v[0, 0]:v[0, 1]]) for i, v in enumerate(l, 0)]
    l_sorted = sorted(l_info, key=lambda x: x[1], reverse=True)

    l_chosen_index = [l_sorted[0][2]]
    l_chosen_unique_bytes = [l_sorted[0][3]]

    LEN_CHOSEN_INDEX = 16
    for _, _, i, unique_bytes in l_sorted:
        print("unique_bytes: {}".format(unique_bytes))

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

    sys.exit()

    a=l[1]
    idxs_lt = np.where(a[:-1, 0]<a[1:, 1])[0]
    print("idxs_lt: {}".format(idxs_lt))
    l_compressed_content = []

    sys.exit()


    for round_nr in range(1, 100):
        print("round_nr: {}".format(round_nr))
        d_arr_comb, d_arr_comb_unique = create_dict_word_count_for_arr(arr, max_byte_length=10)

        l_data = []
        for byte_amount in range(2, 11):
            # print("byte_amount: {}".format(byte_amount))
            arr_comb = d_arr_comb[byte_amount]
            # for unique_bytes in d_arr_comb_unique[byte_amount]['u']:
            for unique_bytes in d_arr_comb_unique[byte_amount]['u'][:50]:
                # check, if shifting/rolling the array is somewhere equal or not
                is_somewhere_equal = False
                unique_bytes_cpy = unique_bytes.copy()
                for i in range(0, byte_amount-1):
                    unique_bytes_cpy = np.roll(unique_bytes_cpy, 1)
                    if np.all(unique_bytes_cpy==unique_bytes):
                        is_somewhere_equal = True
                        break

                if is_somewhere_equal:
                    # print("somewhere equal for: unique_bytes: {}".format(unique_bytes))
                    continue

                idxs_nr_u = np.where(np.all(arr_comb==unique_bytes, axis=1))[0]
                diff = np.diff(np.hstack(((0, ), idxs_nr_u)))
                max_bits = len(bin(np.max(diff))[2:])
                l_diff_bits = [len(bin(i))-2 for i in diff]
                sum_bits_compressed = 4+byte_amount*8+2*8+4+max_bits*len(l_diff_bits)+sum(l_diff_bits)
                sum_bits_raw = byte_amount*8*len(diff)

                l_data.append(((sum_bits_raw-sum_bits_compressed)/sum_bits_raw*100, sum_bits_raw, sum_bits_compressed, unique_bytes.shape[0], unique_bytes.tolist()))

        l_data = sorted(l_data, key=lambda x: (x[0], x[1], x[2]), reverse=True)
        # l_data = sorted(l_data, key=lambda x: (x[1]-x[2])*x[3], reverse=True)
        print("l_data[:5]:\n{}".format('\n'.join(map(str, l_data[:5]))))

        _, _, _, best_length, best_unique_bytes = l_data[0]
        arr_comb = d_arr_comb[best_length]
        idxs_nr_u = np.where(np.all(arr_comb==best_unique_bytes, axis=1))[0]
        diff = np.diff(np.hstack(((0, ), idxs_nr_u)))
        max_bits = len(bin(np.max(diff))[2:])
        l_diff_bits = [len(bin(i))-2 for i in diff]

        content_compressed = (
            bin(best_length)[2:].zfill(4)+
            ''.join(map(lambda x: bin(x)[2:].zfill(8), best_unique_bytes))+
            '{:016b}'.format(len(diff))+
            bin(max_bits)[2:].zfill(4)+
            ''.join(map(lambda x: bin(x)[2:].zfill(max_bits), l_diff_bits))+
            ''.join(map(lambda x: bin(x)[2:], diff))
        )
        content_compressed_fill = content_compressed+'0'*(8-len(content_compressed)%8)
        arr_compressed = np.sum(np.array(list(map(int, content_compressed_fill)), dtype=np.uint8).reshape((-1, 8))*2**np.arange(7, -1, -1), axis=1)

        l_compressed_content.insert(0, arr_compressed)

        # remove all best_unique_bytes from the arr and start over again!
        arr_new = arr[~np.isin(np.arange(0, arr_comb.shape[0]+best_length-1), (idxs_nr_u.reshape((-1, 1))+np.arange(0, best_length)).reshape((-1, )))]

        arr = arr_new

        bytes_size_compressed_content = sum([a.shape[0] for a in l_compressed_content])+arr.shape[0]
        print("bytes_starting_size: {}".format(bytes_starting_size))
        print("bytes_size_compressed_content: {}".format(bytes_size_compressed_content))

    bytes_size_compressed_content = sum([a.shape[0] for a in l_compressed_content])+arr.shape[0]
    print("bytes_starting_size: {}".format(bytes_starting_size))
    print("bytes_size_compressed_content: {}".format(bytes_size_compressed_content))
    # print("len(content_compressed): {}".format(len(content_compressed)))
    # print("len(content_compressed_fill): {}".format(len(content_compressed_fill)))

    # u2 = d_arr_comb_unique[2]['u'][0]
    # idxs_nr_u2 = np.where(np.all(d_arr_comb[2]==u2, axis=1))[0]
    # diff_2 = np.diff(np.hstack(((0, ), idxs_nr_u2)))
    # max_bits_2 = len(bin(np.max(diff_2))[2:])
    # print("max_bits_2: {}".format(max_bits_2))
    # l_diff_bits_2 = [len(bin(i))-2 for i in diff_2]
    # sum_bits_compressed_2 = 4+10*8+2*8+4+len(bin(max(l_diff_bits_2) ))-2*len(l_diff_bits_2)+sum(l_diff_bits_2)
    # sum_bits_raw_2 = 2*8*len(diff_2)

    # u10 = d_arr_comb_unique[10]['u'][0]
    # idxs_nr_u10 = np.where(np.all(d_arr_comb[10]==u10, axis=1))[0]
    # diff_10 = np.diff(np.hstack(((0, ), idxs_nr_u10)))
    # max_bits_10 = len(bin(np.max(diff_10))[2:])
    # print("max_bits_10: {}".format(max_bits_10))
    # l_diff_bits_10 = [len(bin(i))-2 for i in diff_10]
    # sum_bits_compressed_10 = 4+10*8+2*8+4+len(bin(max(l_diff_bits_10) ))-2*len(l_diff_bits_10)+sum(l_diff_bits_10)
    # sum_bits_raw_10 = 2*8*len(diff_10)

    # print("sum_bits_raw_2: {}, sum_bits_compressed_2: {}".format(sum_bits_raw_2, sum_bits_compressed_2))
    # print("sum_bits_raw_10: {}, sum_bits_compressed_10: {}".format(sum_bits_raw_10, sum_bits_compressed_10))

    """
    the structure of the compression idea:
    n=4 bits...#length  bytes
    n bytes...the content, which is used
    l=2 bytes...#amount of numbers, how much the content is used
    b1=4 bits...max length of the length bits of the length for the diff positions
    b2=x bits...length of bits for the diff positon
    diff position=b2 bits...the actual number, which is the diff of position between two contents 
    
    sum of bits: 4+n*8+2*8+4+l*b1+sum(b2)
    """

    sys.exit()

    length = arr.shape[0]
    
    def find_most_common_combinations(arr):
        length = len(arr)
        counts = defaultdict(int)
        for k in range(2, 16):
            print("k: {}".format(k))
            for i in range(0, length-k):
                a = arr[i:i+k]
                t = tuple(a.tolist())
                counts[t] += 1
            print("len(counts):\n{}".format(len(counts)))
        lst_counts = sorted([(k, v, len(k)*v) for k, v in counts.items()], key=lambda x: (x[2], x[1], x[0]), reverse=True)
        print("lst_counts[:100]: {}".format(lst_counts[:100]))
        globals()['lst_counts'] = lst_counts
        return [v for v, _, _ in lst_counts]


    def get_occurence_dict(arr, min_k, max_k):
        length = len(arr)
        counts = defaultdict(dict)
        for k in range(min_k, max_k+1):
            counts_k = defaultdict(int)
            print("k: {}".format(k))
            for i in range(0, length-k):
                a = arr[i:i+k]
                t = tuple(a.tolist())
                counts_k[t] += 1
                # counts[t] += 1
            print("len(counts_k): {}".format(len(counts_k)))
            counts[k] = counts_k
        return counts

    # max_k = 10
    # counts = get_occurence_dict(arr, max_k)
    # lst_t = [k for k in counts]

    # dict_lens_t = defaultdict(list)
    # for t in lst_t:
    #     dict_lens_t[len(t)].append(t)

    # lens = sorted(dict_lens_t.keys())
    # for l in lens:
    #     print("l: {}, len(dict_lens_t[l]): {}".format(l, len(dict_lens_t[l])))

    # occurences = defaultdict(int)
    # amount_splits = defaultdict(int)

    # for k, l1 in enumerate(lens[:-1], 1):
    #     for l2 in lens[k:]:
    #         print("l1: {}, l2: {}".format(l1, l2))
    #         lst_t1 = dict_lens_t[l1]
    #         lst_t2 = dict_lens_t[l2]

    #         print("len(lst_t1): {}, len(lst_t2): {}".format(len(lst_t1), len(lst_t2)))

    #         lst_splits = [(t2, [t2[i:i+l1] for i in range(0, l2-l1+1)]) for t2 in lst_t2]
    #         # all_splits = [t for _, lst in lst_splits for t in lst]
    #         count_splits = defaultdict(int)
            
    #         for t2, splits in lst_splits:
    #             for t1 in splits:
    #                 count_splits[t1] += 1
    #             amount_splits[t2] += len(splits)
            
    #         for j, t1 in enumerate(dict_lens_t[l1], 0):
    #             occurences[t1] += count_splits[t1]
            
            # for t in all_splits:
            #     count_splits[t] += 1
            # found_t1 = []
                # if j%1000==0:
                #     print("j: {}".format(j))
                # s = 0
                # for splits in lst_splits:
                # if t1 in count_splits:
                # if t1 in set_splits:
                    # found_t1.append(t1)
                    # occurences[t1] = l2
                    # break
                    # s += splits.count(t1)
                # occurences[t1] = s
            # for t1 in found_t1:
            #     lst_t1.remove(t1)

    # count_occurrences = []
    # for j, t in enumerate(lst_t, 0):
    #     if j%10000==0:
    #         print("j: {}".format(j))
    #     s_occurrences = 0
    #     len_t = len(t)
    #     for t2 in lst_t:
    #         len_t2 = len(t2)
    #         if len_t>=len_t2:
    #             continue
    #         s_occurrences += np.sum([t==t2[i:i+len_t] for i in range(0, len_t2-len_t+1)])
    #         # if s_occurrences > 100:
    #         #     break
    #         if s_occurrences>0:
    #             break
    #     # print("j: {}, s_occurrences: {}".format(j, s_occurrences))
    #     count_occurrences.append(s_occurrences)

    # sys.exit("Test FAIL! 4321")


    # lst_counts = sorted([(k, len(k)*v) for k, v in counts.items()], key=lambda x: (x[1], x[0]), reverse=True)
    # # lst_counts = sorted([(k, v, len(k)*v) for k, v in counts.items()], key=lambda x: (x[2], x[1], x[0]), reverse=True)
    
    # # find the tuple, which is at least occurring in other tuples!


    # all_merged_idxs_ranges = []

    # for num in range(0, 50):
    #     t = lst_counts[num][0]
    #     print("num: {}, t: {}".format(num, t))
        
    #     a = np.array(t)
    #     if np.all(a[:-1]==a[1:]):
    #         continue

    #     k = len(t)
    #     arr_2d = np.vstack(tuple(arr[i:-k+1+i] for i in range(0, k-1))+(arr[k-1:], )).T

    #     idxs = np.sort(np.where(np.all(arr_2d==t, axis=1))[0]).tolist()
    #     delete_idxs = []
    #     i = 1
    #     i_prev = 0
    #     len_idxs = len(idxs)
    #     while i < len_idxs:
    #         if idxs[i]-idxs[i_prev]<k:
    #             delete_idxs.append(i)
    #         else:
    #             i_prev = i
    #         i += 1
    #     for i in reversed(delete_idxs):
    #         idxs.pop(i)

    #     idxs_ranges = [(i, i+k) for i in idxs]
    #     print("- len(idxs_ranges): {}".format(len(idxs_ranges)))

    #     utils_compress_enwik8.check_merged_idxs_ranges(idxs_ranges)

    #     all_merged_idxs_ranges_new = utils_compress_enwik8.do_merge_idxs_ranges(all_merged_idxs_ranges, idxs_ranges)
    #     print("- len(all_merged_idxs_ranges_new): {}".format(len(all_merged_idxs_ranges_new)))

    #     try:
    #         utils_compress_enwik8.check_merged_idxs_ranges(all_merged_idxs_ranges_new)
    #     except:
    #         idxs_ranges_1 = all_merged_idxs_ranges
    #         idxs_ranges_2 = idxs_ranges
    #         dm = DotMap()
    #         dm.idxs_ranges_1 = idxs_ranges_1
    #         dm.idxs_ranges_2 = idxs_ranges_2
    #         with gzip.open('obj.pkl.gz', 'wb') as f:
    #             dill.dump(dm, f)
    #         sys.exit(-123123123)

    #     all_merged_idxs_ranges = all_merged_idxs_ranges_new

    # diffs = []
    # t = all_merged_idxs_ranges[0]
    # if t[0]>0:
    #     diffs.append(t[0])

    # for t1, t2 in zip(all_merged_idxs_ranges[:-1], all_merged_idxs_ranges[1:]):
    #     diffs.append(t2[0]-t1[1])

    # print("- np.max(diffs): {}".format(np.max(diffs)))

    # u, c = np.unique(diffs, return_counts=True)
    # print("- u.shape: {}".format(u.shape))

    # # get a stat, which tuple is used and how many
    # count_used_combinations = defaultdict(int)
    # for i1, i2 in all_merged_idxs_ranges:
    #     t = tuple(arr[i1:i2].tolist())
    #     count_used_combinations[t] += 1
    # print("- len(count_used_combinations): {}".format(len(count_used_combinations)))

    # arr_rest = []
    # for t1, t2 in zip([(0, 0)]+all_merged_idxs_ranges, all_merged_idxs_ranges+[(length, length)]):
    #     arr_rest += arr[t1[1]:t2[0]].tolist()
    # arr_rest = np.array(arr_rest)

    # print("- arr_rest.shape: {}".format(arr_rest.shape))

    # sys.exit(0)


    print("Calc 'counts_total_length'.")
    min_k = 2
    max_k = 15
    counts = get_occurence_dict(arr, min_k, max_k)
    counts_total_length = {k: len(k)**1.3*v for counts_k in counts.values() for k, v in counts_k.items()}
    # counts_total_length = {k: len(k)**1.3*v for counts_k in counts.values() for k, v in counts_k.items()}
    counts_lists = {k: sorted([(k1, v1) for k1, v1 in v.items()], key=lambda x: (x[1], x[0]), reverse=True) for k, v in counts.items()}
    lst_t = []
    lst_t_len = []
    used_t = defaultdict(int)
    length_lst = arr.shape[0]-max_k
    i = 0
    while i < length_lst:
        if i % 100000 < max_k:
            print("i: {}".format(i))
        lst_t_temp = [tuple(arr[i:i+j].tolist()) for j in range(min_k, max_k+1)]
        # amounts = [(i, counts_total_length[t], used_t[t] if t in used_t else 0) for i, t in enumerate(lst_t_temp)]
        # idx_max = sorted(amounts, key=lambda x: (x[2], x[1], x[0]), reverse=True)[0][0]
        
        # for t in lst_t_temp:
        #     if t in used_t and 

        # amounts = [counts[j][t] for j, t in enumerate(lst_t_temp, min_k)]
        amounts = [counts_total_length[t] for t in lst_t_temp]
        idx_max = np.argmax(amounts)

        t = lst_t_temp[idx_max]
        lst_t.append(t)
        l = len(t)
        lst_t_len.append(l)
        used_t[t] += 1
        i += l




    counts_lst_t = defaultdict(int)
    for t in lst_t:
        counts_lst_t[t] += 1

    print("len(counts_lst_t): {}".format(len(counts_lst_t)))
    lens = [len(t) for t in counts_lst_t.keys()]
    print("len(lens): {}".format(len(lens)))
    u_l, c_l = np.unique(lens, return_counts=True)
    print("u_l: {}".format(u_l))
    print("c_l: {}".format(c_l))
    print("np.sum(u_l*c_l): {}".format(np.sum(u_l*c_l)))

    print("len(used_t): {}".format(len(used_t)))


    amount_idx_of_idx_bits_bytes = np.floor(len(bin(max_k-min_k)[2:])*len(lst_t)/8)
    print("amount_idx_of_idx_bits_bytes: {}".format(amount_idx_of_idx_bits_bytes))
    
    amount_bits_needed = [len(bin(i)[2:]) for i in c_l]
    amount_of_lengths_bits = np.unique(lst_t_len, return_counts=True)[1]
    needed_bytes_idx_table = np.floor(np.sum(amount_of_lengths_bits*amount_bits_needed)/8)
    print("needed_bytes_idx_table: {}".format(needed_bytes_idx_table))
    
    amount_bytes_for_amount_bits_for_amount_of_lengths = np.floor(np.max([len(bin(i)[2:]) for i in amount_bits_needed])*(max_k-min_k)/8)
    print("amount_bytes_for_amount_bits_for_amount_of_lengths: {}".format(amount_bytes_for_amount_bits_for_amount_of_lengths))
    
    amount_bytes_for_amount_lst_bytes = np.floor(np.sum(amount_bits_needed)/8)
    print("amount_bytes_for_amount_lst_bytes: {}".format(amount_bytes_for_amount_lst_bytes))

    amount_bytes_for_bytes_table = np.floor(np.sum(c_l*amount_bits_needed)/8)
    print("amount_bytes_for_bytes_table: {}".format(amount_bytes_for_bytes_table))

    minimal_amount_of_needed_bytes = int(needed_bytes_idx_table+amount_bytes_for_amount_bits_for_amount_of_lengths+amount_bytes_for_amount_lst_bytes+amount_bytes_for_bytes_table)
    print("-----------------------------------------")
    print("minimal_amount_of_needed_bytes: {}".format(minimal_amount_of_needed_bytes))
    # u_t, c_t = np.unique(list(used_t.keys()), return_counts=True)
    # print("u_t: {}".format(u_t))
    # print("c_t: {}".format(c_t))

    # TODO:
    # find the most common combination and take it
    # go recursevly each new list/arr for finding the next most common combination etc.

    # # TODO: make the simplest example for compressing the enwik8 file!
    # k = 2
    # counts = defaultdict(int)
    # lst_t = []
    # for i in range(0, length, k):
    #     if i%1000000==0:
    #         print("i: {}".format(i))
    #     t = tuple(arr[i:i+k].tolist())
    #     lst_t.append(t)
    #     counts[t] += 1
    # t_to_idx = {t: i for i, t in enumerate(counts.keys(), 0)}
    # lst_t_idx_tbl = [t_to_idx[t] for t in lst_t]
