#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
from different_combinations import get_permutation_table
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time
from functools import reduce

from PIL import Image

import numpy as np

from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

from utils_modulo_sequences import prettyprint_dict

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

from math import factorial


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


def get_seq_l(n):
    l = np.arange(1, n+1).tolist()
    seq_l = np.zeros((n, ), dtype=np.int)

    acc = n-1
    while len(l) > 0:
        k = l.pop(0)
        acc_count = 0
        while acc_count < k:
            acc = (acc+1)%n
            if seq_l[acc] == 0:
                acc_count += 1
        seq_l[acc] = k

    return seq_l


def get_seq_l_faster(n):
    l = np.arange(0, n).tolist()
    seq_l = np.zeros((n, ), dtype=np.int)

    acc = 0
    for i in range(0, n):
        if seq_l[-1]!=0:
            return [None]
        acc = (acc+i) % len(l)
        k = l.pop(acc)
        seq_l[k] = i+1
        # acc_count = 0
        # while acc_count < k:
        #     acc = (acc+1)%n
        #     if seq_l[acc] == 0:
        #         acc_count += 1
        # seq_l[acc] = k

    return seq_l


if __name__ == "__main__":
    # # n = 9
    # lst_last_num_n = []
    # for n in range(1, 10001):
    #     if n%100==0:
    #         print("n: {}".format(n))
    #     # seq_l = get_seq_l(n)
    #     seq_l = get_seq_l_faster(n)

    #     # print("n: {}, seq_l: {}".format(n, seq_l))
    #     if seq_l[-1]==n:
    #         lst_last_num_n.append(n)

    # print("lst_last_num_n: {}".format(lst_last_num_n))


    # # find where the sum modulo n == 0!
    # lst_modulo_n_zero = []
    # for i in range(1, 1001):    
    #     if i % 100 == 0:
    #         print("i: {}".format(i))

    #     if ((i**2+i)//2)%i==0:
    #         lst_modulo_n_zero.append(i)
    # print("lst_modulo_n_zero: {}".format(lst_modulo_n_zero))


    def get_stacked_runways(n, i_s): # i_s...jumping list!
    # def get_stacked_runways(n, k, permutate=False):
        arr = np.zeros((n, ), dtype=np.int)
        acc = n-1

        for i in i_s:
            acc = (acc+i)%n
            arr[acc] += 1

        return arr

    n_max = 12

    arr_max_amount_stacks_best = np.zeros((n_max, ), dtype=np.int)
    
    for i_iter in range(0, 1):
        print("i_iter: {}".format(i_iter))
        lst_amount_stacks = []
        lst_max_amount_per_stack = []
        lst_max_amount_stacks_count = []
        lst_max_amount_stacks = []
        lst_i =  []

        for i in range(11, n_max+1):
            lst_i.append(i)
            # print("i: {}".format(i))
            
            # arr = get_stacked_runways(i, i)
            # print("- arr: {}".format(arr))

            lst_arrs_amount_stack = []
            permutation_tbl = get_permutation_table(i)+1
            # for j in range(0, i**3):
            # for j in range(0, factorial(i)//i):
            for i_s in permutation_tbl:
                arr_perm = get_stacked_runways(i, i_s)
                # arr_perm = get_stacked_runways(i, i, permutate=True)
                lst_arrs_amount_stack.append(np.sum(arr_perm>0))
                # print("- j: {}, arr_perm: {}".format(j, arr_perm))
            # print("j: {}, lst_arrs_amount_stack: {}".format(j, lst_arrs_amount_stack))
            arr_arrs_amount_stacks = np.array(lst_arrs_amount_stack)
            max_amount_stack = np.max(arr_arrs_amount_stacks)
            lst_max_amount_stacks.append(max_amount_stack)
            lst_max_amount_stacks_count.append(np.sum(arr_arrs_amount_stacks==max_amount_stack))
            print("lst_i: {}".format(lst_i))
            print("lst_max_amount_stacks: {}".format(lst_max_amount_stacks))
            print("lst_max_amount_stacks_count: {}".format(lst_max_amount_stacks_count))
            # print("i: {}, lst_arrs_amount_stack: {}".format(i, lst_arrs_amount_stack))
            # print("j: {}, max_amount_stack: {}".format(j, max_amount_stack))
            # input("ENTER...")
            arr = get_stacked_runways(i, np.arange(1, i+1))
            lst_amount_stacks.append(np.sum(arr>0))
            lst_max_amount_per_stack.append(np.max(arr))

        # print("n_max: {}".format(n_max))
        # print("lst_amount_stacks: {}".format(lst_amount_stacks))
        # print("lst_max_amount_per_stack: {}".format(lst_max_amount_per_stack))
        # print("lst_max_amount_stacks: {}".format(lst_max_amount_stacks))

        # arr_amount_stacks = np.array(lst_amount_stacks)
        arr_max_amount_per_stack = np.array(lst_max_amount_per_stack)
        arr_max_amount_stacks = np.array(lst_max_amount_stacks)

        idxs = arr_max_amount_stacks_best<arr_max_amount_stacks
        arr_max_amount_stacks_best[idxs] = arr_max_amount_stacks[idxs]

        print("arr_max_amount_stacks_best: {}".format(arr_max_amount_stacks_best))
