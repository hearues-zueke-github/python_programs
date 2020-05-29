#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

# TODO: need to be python convension confirm! (coding standard as much as possible!)

import datetime
import dill
import gzip
import inspect
import os
import pdb
import shutil
import string
import sys

from copy import deepcopy

import numpy as np
from array2gif import write_gif

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization


if __name__ == "__main__":
    def gcd(a, b):
        if a==b:
            return a
        if a<b:
            a, b = b, a
        while b>0:
            t = a%b
            a = b
            b = t
        return a


    def is_int_sqrt(n):
        x_prev = n
        x_now = (n//1+1)//2
        while x_now<x_prev:
            t = (n//x_now+x_now)//2
            x_prev = x_now
            x_now = t
        return x_now**2==n


    def int_sqrt(n):
        x_prev = n
        x_now = (n//1+1)//2
        while x_now<x_prev:
            t = (n//x_now+x_now)//2
            x_prev = x_now
            x_now = t
        return x_now


    max_n = 300
    # l_triples_squares = []
    # for a in range(1, max_n):
    #     a_2 = a**2
    #     for b in range(a, max_n):
    #         div_ab = gcd(a, b)
    #         a_2_b_2 = a_2+b**2
    #         for c in range(b, max_n):
    #             if gcd(div_ab, c)>1:
    #                 continue
    #             if is_int_sqrt(a_2_b_2+c**2):
    #                 l_triples_squares.append((a, b, c))
    
    # for c in range(1, max_n):
    #     c_2 = c**2
    #     for b in range(1, c+1):
    #         div_bc = gcd(b, c)
    #         c_2_b_2 = c_2+b**2
    #         for a in range(1, b+1):
    #             if gcd(div_bc, a)>1:
    #                 continue
    #             if is_int_sqrt(c_2_b_2+a**2):
    #                 l_triples_squares.append((a, b, c))

    # l_sqrt_lens = []
    n_max = 100

    l = []
    l_triples = []
    l_lens_new = []
    len_l = 0
    l_numbers = []
    l_factors_sum = []

    # # 2 square numbers
    # l_doubles = []
    # for i in range(0, n_max+1):
    #     half = i//2+1
    #     for a in range(1, half):
    #         b = i-a
    #         # if gcd(a, b)>1:
    #         #     continue
    #         t = (a, b)
    #         l.append(t)
    #         s = a**2+b**2
    #         sq = int_sqrt(s)
    #         if sq**2==s:
    #             l_numbers.append(sq)
    #             l_doubles.append(t)
    #             l_factors_sum.append(a+b)
    #     # t = len(l)
    #     # l_lens_new.append(t-len_l)
    #     # len_l = t
    # print("l: {}".format(l))
    # print("l_doubles: {}".format(l_doubles))
    # l_unique_nums_doubles = np.unique(list(map(lambda x: np.sum(x), l_doubles))).tolist()
    # # print("l_unique_nums_doubles: {}".format(str(l_unique_nums_doubles).replace(',', '')))
    # print("l_unique_nums_doubles: {}".format(l_unique_nums_doubles))
    # print("sorted(set(l_numbers)): {}".format(sorted(set(l_numbers))))
    # print("sorted(set(l_factors_sum)): {}".format(sorted(set(l_factors_sum))))
    # sys.exit(0)

    # 3 square numbers
    d_pairs = {}
    for i in range(0, n_max+1):
        l_pairs = []
        d_pairs[i] = l_pairs
        third = i//3+1
        for a in range(1, third):
            d1 = i-a
            for b in range(a, third+1):
                c = d1-b
                if c<b:
                    break
                # if gcd(gcd(a, b), c)>1:
                #     continue

                t = (a, b, c)
                l.append(t)
                s = a**2+b**2+c**2
                sq = int_sqrt(s)
                if sq**2==s:
                    l_numbers.append(sq)
                    l_triples.append(t)
                    l_pairs.append(t)
                    l_factors_sum.append(a+b+c)
        t = len(l)
        l_lens_new.append(t-len_l)
        len_l = t

        if len(l_pairs)==0:
            del d_pairs[i]
    print("sorted(set(l_numbers)): {}".format(sorted(set(l_numbers))))
    print("sorted(set(l_factors_sum)): {}".format(sorted(set(l_factors_sum))))

    # # print("l_triples: {}".format(l_triples))
    # l_unique_nums_triples = np.unique(list(map(lambda x: np.sum(x), l_triples))).tolist()
    # # print("unique_sums: {}".format(str(l_unique_nums_triples).replace(',', '')))
    # print("l_unique_nums_triples: {}".format(l_unique_nums_triples))# print("unique_sums: {}".format(str(np.unique(list(map(lambda x: np.sum(x), l_triples))).tolist()).replace(',', '')))

    # # 4 square numbers
    # d_pairs = {}
    # for i in range(1, n_max+1):
    #     print("i: {}".format(i))
    #     l_pairs = []
    #     d_pairs[i] = l_pairs
    #     quatro = i//4+1
    #     for a in range(1, quatro):
    #         a_sq = a**2
    #         d1 = i-a
    #         for b in range(a, quatro):
    #             d2 = d1-b
    #             ab_sq = a_sq+b**2
    #             for c in range(b, quatro+1):
    #                 d = d2-c
    #                 if d<c:
    #                     break
    #                 if gcd(gcd(a, b), c)>1:
    #                     continue

    #                 t = (a, b, c, d)
    #                 l.append(t)
    #                 if is_int_sqrt(ab_sq+c**2+d**2):
    #                     l_triples.append(t)
    #                     l_pairs.append(t)
    #     t = len(l)
    #     l_lens_new.append(t-len_l)
    #     len_l = t

    #     if len(l_pairs)==0:
    #         del d_pairs[i]

    # print("l_triples: {}".format(l_triples))
    l_unique_nums_triples = np.unique(list(map(lambda x: np.sum(x), l_triples))).tolist()
    # print("unique_sums: {}".format(str(l_unique_nums_triples).replace(',', '')))
    print("l_unique_nums_triples: {}".format(l_unique_nums_triples))# print("unique_sums: {}".format(str(np.unique(list(map(lambda x: np.sum(x), l_triples))).tolist()).replace(',', '')))

    sys.exit(0)

    l = sorted(list(map(lambda x: (x, x[0]+x[1]+x[2]), l_triples_squares)), key=lambda x: (x[1], x[0]))
    l_flatten = [i for t in l for i in t[0]]
    print("l_flatten[:20]: {}".format(l_flatten[:20]))
    sys.exit(0)

    """
    2, 1, 2, 3, 2, 4, 2, 1, 2, 4, 8, 3
    [((2, 3, 6), 11),
     ((1, 4, 8), 13),
     ((2, 6, 9), 17),
     ((3, 4, 12), 19),
     ((2, 5, 14), 21),
     ((4, 6, 12), 22),
     ((2, 10, 11), 23),
     ((1, 6, 18), 25),
     ((2, 8, 16), 26),
     ((4, 5, 20), 29),
     ((8, 9, 12), 29),
     ((3, 6, 22), 31),

    [((2, 3, 6), 11),
     ((1, 4, 8), 13),
     ((2, 6, 9), 17),
     ((3, 4, 12), 19),
     ((2, 5, 14), 21),
     ((4, 6, 12), 22),
     ((2, 10, 11), 23),
     ((1, 6, 18), 25),
     ((2, 8, 16), 26),
     ((4, 5, 20), 29),
     ((8, 9, 12), 29),
     ((3, 6, 22), 31),
    """


    # l_triples = []
    # max_n = 10000
    # for a in range(1, max_n):
    #     if a%100==0:
    #         print("a: {}".format(a))
    #     for b in range(a+1, max_n):
    #         if not is_int_sqrt(a**2+b**2):
    #             continue
    #         for c in range(b+1, max_n):
    #             if not is_int_sqrt(a**2+c**2) or not is_int_sqrt(b**2+c**2):
    #                 continue
    #             print("a: {}, b: {}, c: {}".format(a, b, c))
    #             l_triples.append((a, b, c))
    #             break
    #         break

    # print("l_triples: {}".format(l_triples))


        # t1 = a
        # t2 = b

    d_triples_low_top = {}
    d_triples_high_bottom = {}
    # d_triples = {}
    for m in range(2, 5000+1):
        print("m: {}".format(m))
        for n in range(1, m):
            a = m**2-n**2
            b = 2*m*n
            # c = m**2+n**2

            # t1 = a
            # t2 = b
            # while t2>0:
            #     t = t1%t2
            #     t1 = t2
            #     t2 = t
            # if t1>1:
            #     continue
            # d_triples[(a, b) if a<b else (b, a)] = c

            k1, k2 = (a, b) if a<b else (b, a)
            if not k1 in d_triples_low_top:
                d_triples_low_top[k1] = []
            d_triples_low_top[k1].append(k2)
            if not k2 in d_triples_high_bottom:
                d_triples_high_bottom[k2] = []
            d_triples_high_bottom[k2].append(k1)

    l_triples_found = []
    for a, l_b in d_triples_low_top.items():
        if len(l_b)==1:
            continue
        for b in l_b[:-1]:
            if not b in d_triples_low_top:
                continue
            l_c = d_triples_low_top[b]
            if len(l_c)==1:
                continue
            for c in l_c:
                l_ab = d_triples_high_bottom[c]
                if a in l_ab and b in l_ab:
                    print("a: {}, b: {}, c: {}".format(a, b, c))
                    l_triples_found.append((a, b, c))

    # d1 = {k: v for k, v in d_triples_low_top.items() if len(v)>1}
    # d2 = {k2: d_triples_high_bottom[k2] for k, v in d1.items() for k2 in v if len(d_triples_high_bottom[k2])>1}
    # d3 = {k2: d_triples_low_top[k2] for k, v in d2.items() for k2 in v if len(d_triples_low_top[k2])>1}
    # d4 = {k2: d_triples_high_bottom[k2] for k, v in d3.items() for k2 in v if len(d_triples_high_bottom[k2])>1}
    # d5 = {k2: d_triples_low_top[k2] for k, v in d4.items() for k2 in v if len(d_triples_low_top[k2])>1}


    # print("d_triples: {}".format(d_triples))
    # print("len(d_triples): {}".format(len(d_triples)))
