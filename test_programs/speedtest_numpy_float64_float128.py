#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.8.6 -i

# -*- coding: utf-8 -*-

import numpy as np

from pprint import pprint
from time import time

def f1(arr, n):
    for _ in range(0, n):
        arr = (arr**2 + arr * 2 + 1) % 100
    return arr


def time_function(f, args):
    start_time = time()
        
    ret_val = f(*args)

    end_time = time()

    diff_time = end_time - start_time

    return diff_time, ret_val


if __name__ == '__main__':
    print('Hello World!')

    arr_f64 = np.random.randint(0, 1000, (200, 200)).astype(np.float64)
    arr_f128 = arr_f64.copy().astype(np.float64)

    # n = 100
    l_l_diff_time_1 = []
    l_l_diff_time_2 = []
    for n in [10, 50, 100]:
        l_diff_time_1 = []
        l_diff_time_2 = []
        for _ in range(0, 10):
            diff_time_1, ret_val_1 = time_function(f1, (arr_f64, n))
            diff_time_2, ret_val_2 = time_function(f1, (arr_f128, n))

            print("n: {}".format(n))
            print("- diff_time_1: {}".format(diff_time_1))
            print("- diff_time_2: {}".format(diff_time_2))
            
            l_diff_time_1.append(diff_time_1)
            l_diff_time_2.append(diff_time_2)
        
        l_l_diff_time_1.append(l_diff_time_1)
        l_l_diff_time_2.append(l_diff_time_2)

    l_stats_diff_time_1 = [(np.mean(l), np.std(l)) for l in l_l_diff_time_1]
    l_stats_diff_time_2 = [(np.mean(l), np.std(l)) for l in l_l_diff_time_2]

    print('l_stats_diff_time_1:')
    pprint(l_stats_diff_time_1)

    print('l_stats_diff_time_2:')
    pprint(l_stats_diff_time_2)
