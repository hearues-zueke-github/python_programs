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

class SBox(Exception):
    def __init__(self, sbox, bits=8):
        assert bits == 8 or bits == 16
        self.sbox = np.array(sbox)
        self.bits = bits
        self.length = self.sbox.shape[0]
        if bits == 8:
            self.check_sbox_8bit()
        elif bits == 16:
            self.check_sbox_16bit()

    def check_sbox_8bit(self):
        sbox = self.sbox
        if sbox.shape[0] != 256:
            self.sbox_ok = False
            return

        if np.any(sbox == np.arange(0, 256)):
            self.sbox_ok = False
            return

        self.sbox_ok = True

    def check_sbox_16bit(self):
        pass

class SBoxes(Exception):
    def __init__(self, number, sbox_amount, bits=8):
        self.number = number
        self.sbox_amount = sbox_amount
        self.bits = bits

        self.sboxes = self.calculate_sboxes(number, sbox_amount, bits)

    def calculate_sboxes(self, number, sbox_amount, bits=8):
        assert bits == 8 or bits == 16
        # sbox_amount = 256
        sboxes = [[] for _ in range(0, sbox_amount)]

        arr_str = np.array(list((lambda s: s[2:(lambda n: n-n%2)(len(s))])(hex(number)))).reshape((-1, 2)).T
        arr_hex_str = (lambda arr: np.core.defchararray.add("0x", np.core.defchararray.add(arr[0], arr[1])))(arr_str)
        arr = np.vectorize(lambda x: int(x, 16))(arr_hex_str)

        sbox_byte_counter = np.zeros((256, ), dtype=np.int)
        if bits == 16:
            sbox_byte_counter = np.zeros((256*256, ), dtype=np.int)
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

        return [SBox(np.array(sbox)) for sbox in sboxes]

    def get_only_ok_sboxes(self):
        return np.array([sbox for sbox in self.sboxes if sbox.sbox_ok])


if __name__ == "__main__":
    sboxes = SBoxes(3**300000, 200)
