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

from math import factorial as fac

from dotmap import DotMap
from PIL import Image, ImageFont, ImageDraw

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

sys.path.append(PATH_ROOT_DIR+"/..")
import utils_all
import utils_serialization

if __name__=='__main__':
    bits = 4
    sbox_ident = np.arange(0, 2**bits).astype(np.uint16)
    sbox = np.roll(sbox_ident, 1)
    len_sbox = sbox.shape[0]
    print("sbox:\n{}".format(sbox))

    sbox1 = sbox.copy()
    # sbox1_prev = sbox.copy()
    for i in range(0, 1000):
        sbox1_prev = sbox1.copy()
        sbox1 = np.argsort(np.cumsum(sbox1)%len_sbox)
        # print("i: {}, sbox1:\n{}".format(i, sbox1))
        idxs = sbox1==sbox_ident
        idxs_ident = np.where(idxs)[0]
        if idxs_ident.shape[0]>0:
            print("i: {}, idxs_ident: {}".format(i, idxs_ident))
            if idxs_ident.shape[0]==1:
                # print('Only 1 ident position found!')
                idx = idxs_ident[0]
                for j in range(0, len_sbox):
                    k = sbox1_prev[j]
                    if k!=idx:
                        # change with this position!
                        sbox1[j], sbox1[k] = sbox1[k], sbox1[j]
            else:
                sbox1[idxs_ident] = sbox1[np.roll(idxs_ident, 1)]

    sys.exit()

    # create a simple mask array, e.g. a 4 bit number with all numbers split up in binary!
    bits_mask = 4
    mask = [(lambda l: [0]*(bits_mask-len(l))+l)(list(map(int, bin(i)[2:]))) for i in range(0, 2**bits_mask)]
    mask = np.array(mask, dtype=np.uint8).flatten()
    print("mask: {}".format(mask))
    mask = np.random.permutation(mask)
    print("mask permutation: {}".format(mask))


    # now apply this mask to the sbox above with the 2-rot algorithm

    rounds = 2
    idx_mask = 0
    len_mask = mask.shape[0]
    for r in range(0, rounds):
        for i1 in range(0, len_sbox):
            for j in range(1, len_sbox):
                idx_mask = (idx_mask+1)%(len_mask)
                if idx_mask==0:
                    mask = np.roll(mask, 1)
                i2 = (i1+1)%len_sbox
                if mask[idx_mask]==1:
                    if sbox[i1]==j or sbox[j]==i1:
                        continue
                    sbox[i1], sbox[j] = sbox[j], sbox[i1]
        print("r: {}, sbox:\n{}".format(r, sbox))
    # for r in range(0, rounds):
    #     for i in range(0, 2**bits):
    #         for j in range(i+1, 2**bits):
    #             for k in range(i+1, 2**bits):
    #                 idx_mask = (idx_mask+1)%(len_mask)
    #                 # if idx_mask==0:
    #                 #     mask = np.roll(mask, 1)
    #                 if mask[idx_mask]==1:
    #                     if sbox[j]==i or sbox[k]==j or sbox[i]==k:
    #                         continue
    #                     sbox[i], sbox[j], sbox[k] = sbox[j], sbox[k], sbox[i]
    print("sbox: {}".format(sbox))