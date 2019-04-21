#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import numpy as np

from time import time

from dotmap import DotMap

from copy import deepcopy

from indexed import IndexedOrderedDict


def timing_function(f, args):
    start_time = time()
    f(args)
    return time()-start_time


if __name__ == "__main__":
    d = {'a': 0, 'b': '', 'c': []}
    dm = DotMap({'a': 0, 'b': '', 'c': []})
    iod = IndexedOrderedDict({'a': 0, 'b': '', 'c': []})

    def do_dotmap(dm):
        for i in range(0, 100000):
            dm.a += 1
            dm.b += '0'
            dm.c.append(0)

    def do_dict(d):
        for i in range(0, 100000):
            d['a'] += 1
            d['b'] += '0'
            d['c'].append(0)

    dts_dotmap = []
    dts_dotmap_as_dict = []
    dts_dict = []
    dts_idx_ord_dict = []
    for i in range(0, 10):
        print("i: {}".format(i))
        dm_c = deepcopy(dm)
        dt = timing_function(do_dotmap, dm_c)
        dts_dotmap.append(dt)

        dm_c = deepcopy(dm)
        dt = timing_function(do_dict, dm_c)
        dts_dotmap_as_dict.append(dt)

        d_c = deepcopy(d)
        dt = timing_function(do_dict, d_c)
        dts_dict.append(dt)

        iod_c = deepcopy(iod)
        dt = timing_function(do_dict, iod_c)
        dts_idx_ord_dict.append(dt)

    arr_dotmap = np.array(dts_dotmap)
    arr_dotmap_as_dict = np.array(dts_dotmap_as_dict)
    arr_dict = np.array(dts_dict)
    arr_idx_ord_dict = np.array(dts_idx_ord_dict)

    print("arr_dotmap:\n{}".format(arr_dotmap))
    print("arr_dotmap_as_dict:\n{}".format(arr_dotmap_as_dict))
    print("arr_dict:\n{}".format(arr_dict))
    print("arr_idx_ord_dict:\n{}".format(arr_idx_ord_dict))

    print("Mean time with dotmap dot operator:\n  {:.5f} s".format(np.mean(arr_dotmap)))
    print("Mean time with dotmap dict key:\n  {:.5f} s".format(np.mean(arr_dotmap_as_dict)))
    print("Mean time with dict key:\n  {:.5f} s".format(np.mean(arr_dict)))
    print("Mean time with index ordered dict key:\n  {:.5f} s".format(np.mean(arr_idx_ord_dict)))

    # print("dt: {}".format(dt))
    # print("dm_c: {}".format(dm_c))
    # print("dm: {}".format(dm))
