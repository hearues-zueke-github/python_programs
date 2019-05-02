#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

import numpy as np

from math import factorial as fac

import decimal
places = 200
decimal.getcontext().prec = places
from decimal import Decimal as Dec

import matplotlib.pyplot as plt

def get_numbers_list(nums_str):
    return list(map(lambda x: int(x), list(nums_str)))


def convert_to_dec(nums_lst):
    return Dec("0."+"".join(list(map(lambda x: str(x), nums_lst))))


def do_continued_fractions(nums):
    length = len(nums)
    # s = Dec(nums[-1]/Dec(length+1) if nums[-1] > 0 else 1/Dec(length+1))
    s = Dec(nums[-1] if nums[-1] > 0 else 1)
    # s = Dec(length*nums[-1]**length if nums[-1] > 0 else length*1)
    nums_reverse = nums[::-1][1:]
    for n, i in zip(nums[::-1], range(len(nums)-1, -1, -1)):
        # s = 1 / (n/Dec(i+1)+1/s)
        # s = 1 / (n/Dec(i)+1/s)
        s = (n+1/s)
        # s = 1 / (i*n**i+1/s)
    return 1/s


def get_random_dec():
    return Dec("0."+"".join([str(np.random.randint(0, 10**6)) for _ in range(0, places//6+1)][:places]))


if __name__ == "__main__":
    # numbers = np.zeros((100, ), dtype=np.int64)
    # numbers[-1] = 1
    # d_num_str = "0."+"".join(list(map(lambda x: str(x), numbers)))

    # find best fitting decimals!
    nums = [0 for _ in range(0, places)]

    # Trott constant!

    # trott constants:
    # d: 0.20219082524563062081590909090909090909090909090907161311251131408130811111122113412909090909090909090999000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

    # it = 0
    last_best_nums = np.array([0, 0, 0, 0])
    it = 0
    # for it in range(0, 100):
    while it < 100:
        diff_min = 1
        best_nums = np.array([0, 0, 0, 0])
        for i4 in range(0, 10):
          for i3 in range(0, 10):
           for i2 in range(0, 10):
            for i1 in range(0, 10):
                nums[it] = i1
                nums[it+1] = i2
                nums[it+2] = i3
                nums[it+3] = i4
                d = convert_to_dec(nums)
                s = do_continued_fractions(nums)
                diff = np.abs(d-s)
                if diff_min > diff:
                    diff_min = diff
                    best_nums[:] = [i1, i2, i3, i4]
                    print("it: {}, best_nums: {}".format(it, best_nums))

        if np.all(last_best_nums[1:]==best_nums[:-1]):
            print("Need to lower one number to left!")
            for it2 in range(it-1, -1, -1):
                if nums[it2] > 0:
                    nums[it2] -= 1
                    break
                nums[it2] = 9

        print("it: {}, best_nums: {}".format(it, best_nums))
        nums[it:it+4] = best_nums

        print("nums: {}".format(nums))
        d = convert_to_dec(nums)
        s = do_continued_fractions(nums)    
        print("d: {}".format(d))
        print("s: {}".format(s))

        last_best_nums = best_nums

        it += 1

    d = convert_to_dec(nums)
    s = do_continued_fractions(nums)

    print("d: {}".format(d))
    print("s: {}".format(s))

    sys.exit()

    # random search!
    diff_min = 1
    best_d = 0
    best_s = 0
    for i in range(1, 100001):
        d = get_random_dec()
        nums = list(map(lambda x: int(x), str(d)[2:]))
        s = do_continued_fractions(nums)

        diff = np.abs(s-d)

        if diff_min > diff:
            diff_min = diff
            best_d = d
            best_s = s
            print("i: {}".format(i))
            print("diff_min: {}".format(diff_min))
            print("best_d: {}".format(best_d))
    print("diff_min: {}".format(diff_min))
    print("best_d: {}".format(best_d))
    print("best_s: {}".format(best_s))
    sys.exit()

    d_num_str = list("0."+"0"*(places))
    d_num_str[2] = "1"
    # d_num_str[3] = "1"
    # d_num_str[4] = "1"
    # d_num_str[5] = "1"
    # d_num_str[-2] = "1"
    # d_num_str[-1] = "1"

    # d_num_str[11] = "5"
    # d_num_str[19] = "2"
    # d_num_str[29] = "1"
    d_num_str = "".join(d_num_str)
    print("d_num_str: {}".format(d_num_str))
    d_num = Dec(d_num_str)
    print("d_num: {}".format(d_num))
    # print("d_num #2: {:0.11f}".format(d_num))

    str_format = "{{:0.{}f}}".format(places)
    # print("nums_digits_str: {}".format(nums_digits_str))
    
    for it in range(1, 1000001):
        print("it: {}".format(it))

        nums_str = str_format.format(d_num)[2:]
        numbers = get_numbers_list(nums_str)
        s_num = do_continued_fractions(numbers)

        # sign = np.random.randint(0, 2)
        # sign = -1 if sign == 0 else 1
        # d_rnd = Dec(np.random.randint(0, 1000000))/10**(places-6)
        # d_num = d_num+(s_num-d_num)/10+sign*d_rnd
        d_num = d_num+(s_num-d_num)/10

        print("s_num:\n{}".format(s_num))
        print("d_num NEW:\n{}".format(d_num))
        # print()
        # input("ENTER..")

    # print("numbers: {}".format(numbers))

    # print("s_num: {}".format(s_num))

    str(s_num)



