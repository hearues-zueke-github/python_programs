#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import marshal
import pickle
import os

import multiprocessing as mp
import numpy as np

from dotmap import DotMap
from multiprocessing import Process, Queue

from time import time

import Utils

if __name__ == "__main__":
    sbox_amount = 256
    # sbox_byte_counter = np.zeros((256, ), dtype=np.int)
    sbox_byte_counter = np.zeros((256*256, ), dtype=np.int)
    sboxes = [[] for _ in range(0, sbox_amount)]

    arr_str = np.array(list((lambda s: s[2:(lambda n: n-n%2)(len(s))])(hex(5**7000000)))).reshape((-1, 2)).T
    arr_hex_str = (lambda arr: np.core.defchararray.add("0x", np.core.defchararray.add(arr[0], arr[1])))(arr_str)
    arr = np.vectorize(lambda x: int(x, 16))(arr_hex_str)

    arr = arr[:-1]*256+arr[1:]

    print("arr:\n{}".format(arr))

    for i in arr:
        if sbox_byte_counter[i] < sbox_amount:
            sboxes[sbox_byte_counter[i]].append(i)
            sbox_byte_counter[i] += 1

    for idx, sbox in enumerate(sboxes):
        print("idx: {}, len(sbox): {}".format(idx, len(sbox)))
        # Utils.pretty_block_printer(sbox, 8, len(sbox))
        print("")
