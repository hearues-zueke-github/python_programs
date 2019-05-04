#! /usr/bin/python3

# -*- coding: utf-8 -*-

import time

import numpy as np

if __name__ == "__main__":
    n = 30000
    arr_1_uint64 = np.random.randint(0, 2**64, (n, ), dtype=np.uint64).reshape((-1, 1))
    arr_2_uint64 = np.random.randint(0, 2**64, (n, ), dtype=np.uint64)
    # print("arr_uint64: {}".format(arr_uint64))
    # print("arr_uint64.shape: {}".format(arr_uint64.shape))
    # print("arr_uint64.dtype: {}".format(arr_uint64.dtype))

    arr_1_obj = arr_1_uint64.astype(object)
    arr_2_obj = arr_2_uint64.astype(object)

    arr_1_str = arr_1_obj.astype(str).astype(object)
    arr_2_str = arr_2_obj.astype(str).astype(object)

    # print("arr_obj: {}".format(arr_obj))
    # print("arr_obj.shape: {}".format(arr_obj.shape))
    # print("arr_obj.dtype: {}".format(arr_obj.dtype))

    print("Doing 'uint64' dtype!")
    start_time = time.time()
    arr_3_uint64 = arr_1_uint64==arr_2_uint64
    end_time = time.time()
    taken_time_uint64 = end_time-start_time

    print("Doing 'object' dtype!")
    start_time = time.time()
    arr_3_obj = arr_1_obj==arr_2_obj
    end_time = time.time()
    taken_time_obj = end_time-start_time

    print("Doing 'str' dtype!")
    start_time = time.time()
    arr_3_str = arr_1_str==arr_2_str
    end_time = time.time()
    taken_time_str = end_time-start_time

    print("taken_time_uint64:\n  {:.06}s".format(taken_time_uint64))
    print("taken_time_obj:\n  {:.06}s".format(taken_time_obj))
    print("taken_time_str:\n  {:.06}s".format(taken_time_str))
