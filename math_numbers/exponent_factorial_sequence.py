#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import sys

import matplotlib.pyplot as plt

import numpy as np

from math import factorial as fac


PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

def convert_n_to_other_base(n, b):
    l = []
    while n>0:
        l.append(n%b)
        n //= b
    return list(reversed(l))


if __name__=='__main__':

    l_best_b = []
    l_best_b_min = []
    l_best_b_max = []
    for n in range(1, 1001):
        digit_sum_min = n
        # digit_sum_max = 0
        b_best = 1
        # n = fac(6)
        print("n: {}".format(n))        

        b_candidates = [1]
        ls = []
        if n>1:
            for b in range(2, n):
                # b = 10
                l = convert_n_to_other_base(n, b)
                # print("b: {}".format(b))
                # print("l: {}".format(l))

                digit_sum = np.sum(l)

                ls.append((n, b, l, digit_sum))
                # print("digit_sum: {}".format(digit_sum))

                # if digit_sum_max<digit_sum:
                    # digit_sum_max = digit_sum
                if digit_sum_min>digit_sum:
                    digit_sum_min = digit_sum
                    b_best = b
                    b_candidates = [b]
                elif digit_sum_min==digit_sum:
                    b_candidates.append(b)
        print("- b_candidates: {}".format(b_candidates))
        # print("- ls: {}".format(ls))

        # print("b_best: {}".format(b_best))
        l_best_b.append(b_best)

        l_best_b_min.append(b_candidates[0])
        l_best_b_max.append(b_candidates[-1])
        # input('ENTER...')
    arr_best_b = np.array(l_best_b)
    print("l_best_b: {}".format(l_best_b))
    print("arr_best_b: {}".format(arr_best_b))
    print()
    print("l_best_b_min: {}".format(l_best_b_min))
    # print("l_best_b_max: {}".format(l_best_b_max))
    # print("b_best: {}".format(b_best))

    plt.figure()

    plt.plot(np.arange(0, len(l_best_b_min)), l_best_b_min, 'b.')

    plt.show(block=False)

    # find the smallest base, where the following equation is true:
    # # n!<b**n
    # l_seq = []

    # b = 1
    # for n in range(1, 101):
    #     fn = fac(n)
    #     while fn>b**n:
    #         b += 1
    #     l_seq.append(b)

    # print("l_seq: {}".format(np.array(l_seq)))
