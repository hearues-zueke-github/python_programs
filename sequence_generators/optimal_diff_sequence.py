#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import os
import sys
import time

import numpy as np

from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

from collections import namedtuple

import utils_sequence

from numpy.lib.recfunctions import append_fields

path_root_dir = os.path.dirname(os.path.abspath(__file__))+"/"

sys.path.append(path_root_dir+"../combinatorics")
from different_combinations import get_all_combinations_repeat

import decimal
from decimal import Decimal as Dec

import matplotlib.pyplot as plt

decimal.getcontext().prec = 300

# class Partition(namedtuple('PartitionVals', 'n m q p')):
#     [...]
class Partition(Exception):
    __slots__ = ('n', 'm', 'q', 'p')

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.q = n//(m-1)
        self.p = n%(m-1)

    def __str__(self):
        return '({n}, {m}, {q}, {p})'.format(n=self.n, m=self.m, q=self.q, p=self.p)


def get_multipliers(l, base): # base 2
    return np.array([base**i for i in range(l-1, -1, -1)], dtype=object)


def get_amount_of_01_sequence(arr):
    assert np.sum(arr==0) > 0
    assert np.sum(arr==1) > 0
    assert np.sum(arr<0)+np.sum(arr>1) == 0
    arr = np.hstack(((-1, ), arr, (-1, )))
    idx = np.where((arr[1:]==arr[:-1])==False)[0]
    amounts = idx[1:]-idx[:-1]
    # print("arr: {}".format(arr))
    # print("idx: {}".format(idx))
    # print("amounts: {}".format(amounts))

    return amounts


def approach_1(n):
    partitions = [] # will have tuples of ((n, m), (m, q, p))
    partitions2 = []
    # q...n//m
    # p...n%m

    # partition go from 0 to 20
    # e.g. if we want 4 partitions, we have 20//3 == 6 and 20%3 == 2
    # which means that we need 2x 7 and 1x 6 diffs
    # in this case we will get the sequence: 7, 7, 6 -> which gives 0, 7, 14, 20
    # we could simplify the diff sequence as: 1, 1, 0
    # also the optimal diff sequence would be: 1, 0, 1
    # which gives the following sequence: 0, 7, 13, 20
    for m in range(2, n+1):
        partitions.append(((n, m), (m, n//(m-1), n%(m-1))))
        partitions2.append(Partition(n, m))
    # n = (q+1)*p+q*(m-1-p)
    # print("partitions:\n{}".format(partitions))


    # # try different fix m, p and changing only p to get n:
    # ns = []
    # m = 8
    # p = 2
    # print("m: {}, p: {}".format(m, p))
    # def get_n(m, q, p):
    #     return (q+1)*p+q*(m-1-p)
    # for q in range(1, 21):
    #     ns.append((q, get_n(m, q, p)))


    lens_max_lengths = []

    all_choosen_amounts_per_row = {}
    all_opt_arrs = {}
    # m = 7
    for m in range(2, 15):
        print("m: {}".format(m))
        arr = get_all_combinations_repeat(2, m-1)
        ps = {}
        for p in range(1, m-1):
            ps[p] = arr[np.sum(arr, axis=1)==p]

        lens_max_lengths_p = []

        arr_opt = np.zeros((m-2, m-1), dtype=np.int)

        # p...1:m-1
        lsts = []
        for p in range(1, m-1):
            print("p: {}".format(p))
            # p = 5
            arr_p = ps[p]
            # print("arr_p:\n{}".format(arr_p))

            amounts_per_row = [get_amount_of_01_sequence(row) for row in arr_p]
            len_per_row = [row.shape[0] for row in amounts_per_row]

            # print("amounts_per_row:\n{}".format(amounts_per_row))
            # print("len_per_row:\n{}".format(len_per_row))


            idxs = np.where(len_per_row==np.max(len_per_row))[0]
            print("idxs: {}".format(idxs))
            print("p: {}, len(idxs): {}".format(p, len(idxs)))
            choosen_amounts_per_row = np.array([amounts_per_row[idx] for idx in idxs])
            print("choosen_amounts_per_row:\n{}".format(choosen_amounts_per_row))
            all_choosen_amounts_per_row[(m, p)] = choosen_amounts_per_row
            len_idxs = len(idxs)
            lens_max_lengths_p.append(len(idxs))

            arr_p = arr_p[idxs]
            # if len_idxs > 1:

            # input("ENTER...")

            arr_p_inverse = arr_p.copy()
            for i in range(0, arr_p.shape[0]):
                arr_p_inverse[i] = arr_p[i][::-1]

            print("arr_p:\n{}".format(arr_p))
            print("arr_p_inverse:\n{}".format(arr_p_inverse))

            # multipliers = np.array([2**i for i in range(arr_p.shape[1]-1, -1, -1)], dtype=object)
            multipliers = get_multipliers(arr_p.shape[1]-1, 2)
            # print("multipliers: {}".format(multipliers))

            sums_p = np.sum(arr_p*multipliers, axis=1)
            sums_inv_p = np.sum(arr_p_inverse*multipliers, axis=1)

            print("sums_p: {}".format(sums_p))
            print("sums_inv_p: {}".format(sums_inv_p))

            diffs = sums_p-sums_inv_p
            print("diffs: {}".format(diffs))

            # arr = np.array([[0, 3], [1, 7], [2, 5], [3, 1]], dtype=object)
            idx = np.argsort(diffs)

            diffs = diffs[idx]
            arr_p = arr_p[idx]
            arr_p_inverse = arr_p_inverse[idx]
            sums_p = sums_p[idx]
            sums_inv_p = sums_inv_p[idx]

            i_opt = np.where(diffs >= 0)[0][0]
            # print("i_opt: {}".format(i_opt))

            diffs = diffs[i_opt:]
            arr_p = arr_p[i_opt:]
            arr_p_inverse = arr_p_inverse[i_opt:]
            sums_p = sums_p[i_opt:]
            sums_inv_p = sums_inv_p[i_opt:]

            # print("diffs:\n{}".format(diffs))
            # print("arr_p:\n{}".format(arr_p))
            # print("sums_p:\n{}".format(sums_p))
            # print("sums_inv_p:\n{}".format(sums_inv_p))

            amount_of_min_diff = np.sum(diffs==diffs[0])
            # print("amount_of_min_diff: {}".format(amount_of_min_diff))

            opt_idx = 0
            arr_p_opt = arr_p[opt_idx]
            if amount_of_min_diff > 1:
                opt_idx = np.argsort(sums_p[:amount_of_min_diff])[0]
                arr_p_opt = arr_p[opt_idx]

            # print("arr_p_opt: {}".format(arr_p_opt))
            # lsts.append((p, arr_p_opt))
            arr_opt[p-1] = arr_p_opt
        lens_max_lengths.append((m, lens_max_lengths_p))
        all_opt_arrs[m] = arr_opt

        # break

    # x = np.array(np.arange(0,10), dtype = [('x', float)]);
    # y = np.array(np.arange(10,20), dtype = [('y', float)])
    # z = append_fields(x, 't', np.array(y, dtype=np.float))
    # print("z:\n{}".format(z))

    # x = np.array(arr[:, 0], dtype=[('a1', object)])
    # y = np.array(arr[:, 1], dtype=[('a2', object)])
    # # z = append_fields(x, 'a2', arr[:, 1])
    # print("x:\n{}".format(x))
    # print("y:\n{}".format(y))



if __name__ == "__main__":
    print("path_root_dir: {}".format(path_root_dir))

    # n = 20
    # approach_1(n)

    dec_1 = Dec(1)
    m = 600 # need m-1 points!
    # dec_m_1 = dec_1 / (m-1)
    # idxs_0_1 = [dec_m_1 * i for i in range(0, m)]

    dec_0 = Dec(0)
    dec_0_5 = Dec(1) / 2

    # print("idxs_0_1: {}".format(idxs_0_1))


    print("m: {}".format(m))
    # n = 31
    # n...Decimal number
    # e.g. n=23.34, m = 3
    # we have ranges: 0<=d<1/3, 1/3<=d<2/3, 2/3<=d<1, where d = n % 1
    def get_decimal_partition_modulo(n, m):
        d = n%dec_1
        return int("{:.1f}".format(d*Dec(m+1)).split(".")[0])

    partitions = 15
    path_images = "images/partitions_{}/".format(partitions+1)
    if not os.path.exists(path_images):
        os.makedirs(path_images)
    else:
        cwd = os.getcwd()
        os.chdir("./"+path_images)
        print("Creating the gif!")
        os.system("rm -r *.png")
        os.chdir(cwd)
    
    max_n = 300
    ns = np.cumsum(np.arange(1, 100))+1
    ns = ns[ns <= max_n+10]
    print("ns: {}".format(ns))
    pixs = []
    files_path = []
    # sys.exit()
    for m in ns:
    # for m in list(range(2, 101, 4))+list(range(102, 301, 8))+list(range(302, 501, 12)):
        dec_m_1 = dec_1 / (m-1)
        idxs_0_1 = [dec_m_1 * i for i in range(0, m)]

        print("m: {}".format(m))
        # diffs_tbl = np.zeros((m-1, m-1), dtype=np.int)
        diffs_partition_tbl = np.zeros((m-2, m-2), dtype=np.int)
        for p in range(m, m*2-2):
        # for p in range(m-1, m*2-2):
            idxs_0_n = [p*idx for idx in idxs_0_1]
            # print("idxs_0_n:\n{}".format(idxs_0_n))

            # idxs_0_n_opt = [i-(i%dec_1) if i % dec_1 < dec_0_5 else i-(i%dec_1)+1 for i in idxs_0_n]
            # idxs_0_n_opt_int = list(map(int, idxs_0_n_opt))
            # print("idxs_0_n_opt_int: {}".format(idxs_0_n_opt_int))

            # arr = np.array(idxs_0_n_opt_int)
            # diffs = arr[1:]-arr[:-1]
            # diffs_0_1 = diffs-diffs[np.argmin(diffs)]
            # diffs_tbl[p-m+1] = diffs_0_1

            arr2 = np.array([get_decimal_partition_modulo(i, partitions) for i in idxs_0_n])
            diffs_partition_tbl[p-m] = arr2[1:-1]#+partitions-arr2[:-1]
            # diffs_partition_tbl[p-m+1] = arr2[1:]+partitions-arr2[:-1]
            
            # print("diffs:\n{}".format(diffs))
            # print("p: {}".format(p))
            # print("diffs_0_1:\n{}".format(diffs_0_1))
            # lst = list(map(str,diffs_0_1))
            # num = int("".join(lst), 2)
            # print("num: {}".format(num))
        # print("diffs_tbl:\n{}".format(diffs_tbl))

        # pix = diffs_tbl.astype(np.uint8)*255
        # img = Image.fromarray(pix)
        # img.show()

        # diffs_partition_tbl
        pix2 = diffs_partition_tbl.astype(np.uint8)*(255//partitions)
        img2 = Image.fromarray(pix2)
        img2 = img2.resize((max_n, max_n), Image.ANTIALIAS)
        # img2.show()

        # img.save("images/partitions_binary_m_{}.png".format(m))
        file_path = path_images+"partitions_parts_{}_m_{:03}.png".format(partitions+1, m)
        img2.save(file_path)
        
        pixs.append(np.array(img2))
        files_path.append(file_path)
        
        if m == ns[0]:
            for it in range(0, 6):
                img2.save(path_images+"partitions_parts_{}_m_{:03}_{:02}.png".format(partitions+1, m, it))
        elif m == ns[-1]:
            for it in range(0, 20):
                img2.save(path_images+"partitions_parts_{}_m_{:03}_{:02}.png".format(partitions+1, m, it))

    # for pix1, pix2, file_path in zip(pixs[:-1], pixs[1:], files_path):
    #     pix1f = pix1.astype(np.float)
    #     pix2f = pix2.astype(np.float)
    #     for i in range(1, 3):
    #         pix = ((pix1f*(3-i)+pix2f*i)/3).astype(np.uint8)
    #         Image.fromarray(pix).save(file_path.replace(".png", "_{:02}.png".format(i)))

    cwd = os.getcwd()
    os.chdir("./"+path_images)
    animated_gif_name_temp = "animated_{:03}.gif"
    animated_mp4_name_temp = "outfile_animated_{:03}.mp4"
    animated_gif_name = animated_gif_name_temp.format(0)
    animated_mp4_name = animated_mp4_name_temp.format(0)
    i = 0
    while os.path.exists(animated_gif_name):
        i += 1
        animated_gif_name = animated_gif_name_temp.format(i)
        animated_mp4_name = animated_mp4_name_temp.format(i)

    print("Creating the gif!")
    os.system("convert -delay 10 loop 0 *.png {animated_gif_name}".format(animated_gif_name=animated_gif_name))
    print("Creating the mp4!")
    os.system('ffmpeg -f gif -i {} {}'.format(animated_gif_name, animated_mp4_name))
    os.chdir(cwd)

    # TODO: NEED TO BE ON MY WEBSITE!!!

    # arr = np.random.randint(0, 2, (n, ))
    # arr = np.zeros((30, ), dtype=np.int)
    # arr[:15] = 1
    # arr = np.random.permutation(arr)
    # print("arr:\n{}".format(arr))
    # arr_rev = arr[::-1]
    # print("arr_rev:\n{}".format(arr_rev))

    # diff = arr-arr_rev
    # print("diff:\n{}".format(diff))
    # diff_abs = np.abs(diff)
    # print("diff_abs:\n{}".format(diff_abs))

    # pos_idx = np.hstack(((0,), np.where(~(arr[:-1]==arr[1:]))[0], arr.shape[0]))
    # print("pos_idx: {}".format(pos_idx))

    # amounts_alternating = pos_idx[1:]-pos_idx[:-1]
    # print("amounts_alternating: {}".format(amounts_alternating))

    # diffs = amounts_alternating-amounts_alternating[::-1]
    # print("diffs: {}".format(diffs))

    # sum_alternating_diff = np.sum(np.abs(diffs))
    # print("sum_alternating_diff: {}".format(sum_alternating_diff))
