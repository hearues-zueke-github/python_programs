#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy
from dotmap import DotMap

# from sortedcontainers import SortedSet

from collections import defaultdict

# sys.path.append("../combinatorics/")
# import different_combinations as combinations
# sys.path.append("../math_numbers/")
# from prime_numbers_fun import get_primes

# from time import time
# from functools import reduce

from os.path import expanduser
PATH_HOME = expanduser("~")+'/'
print("PATH_HOME: {}".format(PATH_HOME))

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style


def time_measure(f, args):
    start_time = time()
    ret = f(*args)
    end_time = time()
    diff_time = end_time-start_time
    return ret, diff_time


class ShowImg(Frame, object):
    def __init__(self, img):
        parent = Tk()
        Frame.__init__(self, parent)
        self.pack(fill=BOTH, expand=1)
        label1 = Label(self)
        label1.photo= ImageTk.PhotoImage(img)
        label1.config(image=label1.photo)
        label1.pack(fill=BOTH, expand=1)
        parent.mainloop()


def get_arr():
    print("Hello World!")
    file_path_enwik8 = PATH_HOME+'Downloads/enwik8'

    print("file_path_enwik8: {}".format(file_path_enwik8))

    if not os.path.exists(file_path_enwik8):
        print("Please make sure, that '{}' does exist!".format(file_path_enwik8))
        sys.exit(-1)

    with open(file_path_enwik8, 'rb') as f:
        data = f.read()

    # data = [1, 2, 3, 4, 5, 6]

    used_length = 1000000
    # used_length = len(data)
    data_str = data.decode('utf-8')[:used_length]
    lst_data = list(data)[:used_length]
    arr = np.array(lst_data, dtype=object)

    print("used_length: {}".format(used_length))

    # combine with a defined length k and look how many 
    return arr


if __name__ == "__main__":
    arr = get_arr()
    
    def find_most_common_combinations(arr):
        length = len(arr)
        # k = 2
        # exponent_arr = np.arange(k-1, -1, -1)
        # num_mults = 256**exponent_arr
        counts = defaultdict(int)
        for k in range(2, 16):
            print("k: {}".format(k))
            for i in range(0, length-k):
                # print("i: {}".format(i))
                a = arr[i:i+k]
                t = tuple(a.tolist())
                counts[t] += 1
            # print("counts:\n{}".format(counts))
            print("len(counts):\n{}".format(len(counts)))
        lst_counts = sorted([(k, v, len(k)*v) for k, v in counts.items()], key=lambda x: (x[2], x[1], x[0]), reverse=True)
        print("lst_counts[:100]: {}".format(lst_counts[:100]))
        globals()['lst_counts'] = lst_counts
        return [v for v, _, _ in lst_counts]


    def get_occurence_dict(arr, max_k):
        length = len(arr)
        counts = defaultdict(int)
        for k in range(2, max_k+1):
            print("k: {}".format(k))
            for i in range(0, length-k):
                a = arr[i:i+k]
                t = tuple(a.tolist())
                counts[t] += 1
            print("len(counts):\n{}".format(len(counts)))
        return counts


    max_k = 17
    counts = get_occurence_dict(arr, max_k)
    lst_counts = sorted([(k, v, len(k)*v) for k, v in counts.items()], key=lambda x: (x[2], x[1], x[0]), reverse=True)
    t = lst_counts[0][0]
    print("t: {}".format(t))
    # t = lst_combinations[0]
    k = len(t)
    arr_2d = np.vstack(tuple(arr[i:-k+1+i] for i in range(0, k-1))+(arr[k-1:], )).T
    
    # if k==2:
    #     arr_2d = np.vstack((arr[:-1], arr[1:])).T
    # else:
    #     sys.exit(-123213213)

    # arr_2d_2 = np.vstack(tuple(arr[i:-k+1+i] for i in range(0, k-1))+(arr[k-1:], )).T

    # print("arr_2d.shape: {}".format(arr_2d.shape))
    # print("arr_2d_2.shape: {}".format(arr_2d_2.shape))
    # assert np.all(arr_2d==arr_2d_2)

    idxs = np.where(np.all(arr_2d==t, axis=1))[0]
    # idxs = np.hstack((0, idxs, arr.shape[0]))
    idxs_ranges = []
    if idxs[0]!=0:
        idxs_ranges.append([0, idxs[0]])
    if idxs[-1]+k<arr.shape[0]:
        idxs_ranges.append([idxs[-1]+k, arr.shape[0]])
    idxs_ranges += [[i+k, j] for i, j in zip(idxs[:-1], idxs[1:]) if i+k<j]

    print("idxs.shape: {}".format(idxs.shape))
    diff = idxs[1:]-idxs[:-1]
    print("np.max(diff): {}".format(np.max(diff)))

    # for i1, i2 in idxs_ranges:
    #     arr_part = arr[i1:i2]
    #     # now commes the recursive part!

    # max_k = 17
    # counts = get_occurence_dict(arr, max_k)
    # lst_combinations = find_most_common_combinations(arr)

    # 

    sys.exit("TEST1")

    print("Calc 'counts_total_length'.")
    counts_total_length = {k: len(k)*v for k, v in counts.items()}
    lst_t = []
    lst_t_len = []
    length = arr.shape[0]-max_k
    i = 0
    while i < length:
        if i % 100000 < max_k:
            print("i: {}".format(i))
        lst_t_temp = [tuple(arr[i:i+j].tolist()) for j in range(2, max_k+1)]
        amounts = [counts_total_length[t] for t in lst_t_temp]
        # amounts = [counts[t] for t in lst_t_temp]
        idx_max = np.argmax(amounts)
        t = lst_t_temp[idx_max]
        lst_t.append(t)
        l = len(t)
        lst_t_len.append(l)
        i += l

    counts_lst_t = defaultdict(int)
    for t in lst_t:
        counts_lst_t[t] += 1

    lens = [len(t) for t in counts_lst_t.keys()]
    print("len(lens): {}".format(len(lens)))
    u, c = np.unique(lens, return_counts=True)
    print("u: {}".format(u))
    print("c: {}".format(c))

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
