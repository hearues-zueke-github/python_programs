#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import numpy as np

def get_koloski_sequence(max_i):
    lst = [1, 2, 2, 1, 1]
    num = 1
    idx = 3

    for i in range(3, max_i):
        amount = lst[idx]
        # print("i: {}, num: {}, amount: {}".format(i, num, amount))
        lst.extend((num+1, )*lst[idx])
        # print("lst:\n{}".format(lst))
        num = (num+1)%2
        idx += 1

    return lst

def get_rng_koloski_sequence(seed, max_i):
    lst = seed
    num = (seed[-1])%2

    idx = 0

    for i in range(0, max_i):
        amount = lst[idx]
        # print("i: {}, num: {}, amount: {}".format(i, num, amount))
        lst.extend((num+1, )*lst[idx])
        # print("lst:\n{}".format(lst))
        num = (num+1)%2
        idx += 1

    return lst


if __name__ == "__main__":
    lst = get_koloski_sequence(1000)
    # print("orig lst:\n{}".format(lst))
    lst_str = "".join(list(map(str, lst)))
    print("orig lst_str:\n{}".format(lst_str))
    
    # lst = get_rng_koloski_sequence([1, 2], 1000)
    # lst_str = "".join(list(map(str, lst)))
    # print("lst_str:\n{}".format(lst_str))

    # print("lst: {}".format(lst))

    # print("Check sequence:")

    length = len(lst)

    arr = np.array(lst)

    arr_splits = [arr[:length-length%i].reshape((-1, i))-1 for i in range(2, 11)]

    arr_nums = [np.sum(arr*2**np.arange(arr.shape[1]-1, -1, -1), axis=1) for arr in arr_splits]

    

    print("arr_nums[2]:\n{}".format(arr_nums[2]))

    # # for arr in arr_splits:
    # for arr in arr_nums:
    #     print("arr:\n{}".format(arr))
    #     print("np.unique(arr):\n{}".format(np.unique(arr)))
    #     print("arr.shape: {}".format(arr.shape))

    # # current_num = 1
    # # length = len(lst)
    # # idx = 0
    # # for i, num in enumerate(lst):
    # #     pass