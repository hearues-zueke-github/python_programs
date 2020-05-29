#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

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

import matplotlib.pyplot as plt

import numpy as np
from array2gif import write_gif

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

from utils_sequence import check_if_prime, num_to_base

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization

sys.path.append(PATH_ROOT_DIR+'/../math_numbers')
from prime_numbers_fun import get_primes


if __name__ == "__main__":
    # l_xy = [
    #  (1, 11),
    #  (3, 23),
    #  (7, 97),
    #  (9, 149),
    #  (13, 2003),
    #  (29, 8419),
    #  (33, 8923),
    #  (79, 31469),
    #  (81, 404671),
    #  (83, 791773),
    #  (93, 1291783),
    #  (97, 1388587),
    #  (113, 16718003),
    #  (127, 26378017),
    #  (133, 55187023),
    #  (163, 58374053),
    #  (173, 115558063),
    #  (229, 133547119),
    #  (241, 1320036131),
    #  (243, 2871858133)
    # ]
    # l_x, l_y = list(map(list, zip(*l_xy)))

    # arr_x = np.array(l_x, dtype=np.int64)
    # arr_y = np.array(l_y, dtype=np.int64)
    
    # # approx function: f(x) = b*a**x
    # a = 2


    # sys.exit()



    # path_primes = '/tmp/primes_5_10_7.pkl.gz'
    # # path_primes = '/tmp/primes_5_10_7.pkl.gz'
    # if not os.path.exists(path_primes):
    #     l_primes = [i for i in get_primes(5*10**7)]
    #     with gzip.open(path_primes, 'wb') as f:
    #         dill.dump(l_primes, f)
    # else:
    #     with gzip.open(path_primes, 'rb') as f:
    #         l_primes = dill.load(f)

    # print("arr.shape: {}".format(arr.shape))
    # print("arr[0]:  {}".format(arr[0]))
    # print("arr[-1]: {}".format(arr[-1]))
    # sys.exit(0)
    path_primes = '/tmp/primes_data_test_2.dat'
    with open(path_primes, 'rb') as f:
        l_primes = np.fromfile(f, dtype=np.uint64)[1:]

    # print("l_primes: {}".format(l_primes))
    print("len(l_primes): {}".format(len(l_primes)))

    # l = []
    # l_length_append = []
    # for i in range(1, 301):
    #     for k in range(1, 10000):
    #         v = int('{}{}'.format(i, k))
    #         if check_if_prime(v, l_primes):
    #             break
    #     l.append(v)
    #     l_length_append.append(len(str(v))-len(str(i)))

    # print("l: {}".format(l))
    # print("l_length_append: {}".format(l_length_append))
    # sys.exit(0)

    max_n = 0
    max_k = 0
    max_concat = 0
    best_n_concat = 0
    l_smallest_concats_primes = []
    l_smallest_concats_extend_primes = []

    # b = 2
    # l = []

    # lx = []
    # ly = []

    argv = sys.argv
    b = int(argv[1])
    n_start = int(argv[2])

    for n in range(n_start, n_start+100000):
    # for n in range(10000000, 10300000):
    # for n in range(2000000, 5000000):
    # for n in range(1, 200000):
    # for n in range(1, 300):
    # for n in range(2405010000000, 2405010003000):
    # for n in range(3000000, 10000000):
        if n%1000==0:
            print("n: {}".format(n))
    # for n in range(1, 10000):
        is_found_prime_for_i = False

        for k in range(1, 10):
            n2 = n*b**k
            for j in range(1, b**k):
                n_concat = n2+j
                if check_if_prime(n_concat, l_primes):
                    is_found_prime_for_i = True
                    concat_part = n_concat-n*b**k
                    break
            # for j1 in range(0, 10**(k-1)):
            #     n3 = n2+j1*10
            #     for j2 in [1, 3, 7, 9]:
            #         n_concat = n3+j2
            #         if check_if_prime(n_concat, l_primes):
            #             is_found_prime_for_i = True
            #             concat_part = n_concat-n*10**k
            #             break
            #     if is_found_prime_for_i:
            #         break
            if is_found_prime_for_i:
                break
        assert is_found_prime_for_i

        if max_k<k:
            l_smallest_concats_primes.append((k, n_concat))
            l_smallest_concats_extend_primes.append((k, concat_part, n_concat))
            max_k = k
            max_n = n
            max_concat = concat_part
            best_n_concat = n_concat
            print("max_k: {}, max_n: {}, best_n_concat: {}".format(max_k, max_n, best_n_concat))
        if max_k==k and max_concat<concat_part:
            l_smallest_concats_extend_primes.append((k, concat_part, n_concat))
            max_concat = concat_part
        # l.append(n_concat)

        # lx.append(n)
        # ly.append(sum([b**i for i in range(1, k-1)])+concat_part)

    # print("l: {}".format(l))
    print("b: {}".format(b))
    print("l_smallest_concats_primes: {}".format(l_smallest_concats_primes))
    l_smallest_concats_primes_base = [(n, num_to_base(v, b)[::-1]) for n, v in l_smallest_concats_primes]
    print("l_smallest_concats_primes_base: {}".format(l_smallest_concats_primes_base))
    print("l_smallest_concats_extend_primes: {}".format(l_smallest_concats_extend_primes))
    
    # plt.figure()
    # plt.title('b: {}, n = [{}, {}]'.format(b, lx[0], lx[-1]))
    # plt.plot(lx, ly, '.b', markersize=4)
    # plt.show(block=False)

    sys.exit()

    for n in range(0, 7):
        max_j = 0
        best_i = 0
        best_n_concat = 0
        for i in range(10**n, 10**(n+1)):
            is_found_prime_for_i = False
            for j in range(1, 6):
                n1 = i*10**j
                if j==1:
                    k1 = 0
                    for k2 in [1, 3, 7, 9]:
                        n_concat = n1+k2
                        if check_if_prime(n_concat, l_primes):
                            if max_j<j:
                                max_j = j
                                best_i = i
                                best_n_concat = n_concat
                            is_found_prime_for_i = True
                            break
                else:
                    for k1 in range(0, 10**(j-1)):
                        n2 = n1+k1*10
                        for k2 in [1, 3, 7, 9]:
                            n_concat = n2+k2
                            if check_if_prime(n_concat, l_primes):
                                if max_j<j:
                                    max_j = j
                                    best_i = i
                                    best_n_concat = n_concat
                                is_found_prime_for_i = True
                                break
                        if is_found_prime_for_i:
                            break
                if is_found_prime_for_i:
                    break

            # if is_found_prime_for_i:
            #     print("i: {}, n_concat: {}".format(i, n_concat))

        print("best_i: {}, max_j: {}, best_n_concat: {}".format(best_i, max_j, best_n_concat))
