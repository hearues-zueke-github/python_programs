#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

from dotmap import DotMap

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

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

if __name__ == "__main__":
    arr = utils_compress_enwik8.get_arr(used_length=-1)
    
    d_arr_comb = {}
    d_arr_comb_unique = {}
    arr1 = arr.copy()

    print('comb_nr: 1')
    arr_comb = arr.copy().reshape((-1, 1))
    d_arr_comb[1] = arr_comb
    arr_comb_view = arr_comb.reshape((-1, )).view(dtype=[('f{}'.format(i), '<u1') for i in range(0, 1)])
    u, c = np.unique(arr_comb_view, return_counts=True)
    d_arr_comb_unique[1] = {'u': u, 'c': c}

    for comb_nr in range(2, 4):
        print("comb_nr: {}".format(comb_nr))
        arr_comb = np.hstack((arr_comb[:-1], arr1[comb_nr-1:].reshape((-1, 1))))
        d_arr_comb[comb_nr] = arr_comb
        arr_comb_view = arr_comb.reshape((-1, )).view(dtype=[('f{}'.format(i), '<u1') for i in range(0, 2)])
        u, c = np.unique(arr_comb_view, return_counts=True)
        d_arr_comb_unique[comb_nr] = {'u': u, 'c': c}

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
