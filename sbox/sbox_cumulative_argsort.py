#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

# TODO: need to be python convension confirm! (coding standard as much as possible!)

import binascii
import datetime
import dill
import gzip
import hashlib
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

def print_program_usage():
    # PROGRAM_USAGE = 'usage: <program> (-h) <seed_num> <path_to_save>'
    print('usage: <program> (-h) <seed_num> <path_to_save>')
    print("If used with '-h', then seed_num can be a hex number too.")

def restore_sbox(sbox, sbox_prev):
    sbox = sbox.copy()
    n = sbox.shape[0]

    idxs = sbox==np.arange(0, sbox.shape[0])
    idxs_ident = np.where(idxs)[0]
    if idxs_ident.shape[0]==0:
        return sbox

    if idxs_ident.shape[0]==1:
        idx = idxs_ident[0]
        for j in range(0, n):
            k = sbox_prev[j]
            if k!=idx:
                sbox[j], sbox[k] = sbox[k], sbox[j]
                return sbox

    len_idxs_ident = idxs_ident.shape[0]
    idxs_ident_ident = np.arange(0, len_idxs_ident)
    is_found_roll = False
    for j in range(0, n-len_idxs_ident+1):
        idxs_roll = np.argsort(sbox_prev[j:j+len_idxs_ident])
        if np.any(idxs_roll==idxs_ident_ident):
            continue

        sbox[idxs_ident] = sbox[idxs_ident[idxs_roll]]
        return sbox

    sbox1_prev_part = np.hstack((sbox_prev[-len_idxs_ident+1:], sbox_prev[:len_idxs_ident-1]))
    for j in range(0, len_idxs_ident-1):
        idxs_roll = np.argsort(sbox1_prev_part[j:j+len_idxs_ident])
        if np.any(idxs_roll==idxs_ident_ident):
            continue

        sbox[idxs_ident] = sbox[idxs_ident[idxs_roll]]
        return sbox

    sbox[idxs_ident] = sbox[np.roll(idxs_ident, 1)]
    return sbox


def get_sha512_hash_2byte_from_num(seed_num):
    seed_num_hex_str = (lambda x: '0'*(len(x)%2)+x)(hex(seed_num)[2:])
    byte_arr = bytearray([int(seed_num_hex_str[2*i:2*(i+1)], 16) for i in range(0, len(seed_num_hex_str)//2)])
    digest = hashlib.sha512(bytearray(byte_arr)).hexdigest()
    l_digest = [int(digest[4*i:4*(i+1)], 16) for i in range(0, 32)]
    return l_digest


def create_master_sbox_2byte(seed_num):
    l_digest_concat = []
    for i in range(0, 2048):
        l_digest = get_sha512_hash_2byte_from_num(seed_num+i)
        l_digest_concat += l_digest
    master_sbox = np.argsort(l_digest_concat)
    master_sbox = restore_sbox(master_sbox, master_sbox)
    return master_sbox


if __name__=='__main__':
    # TODO: create a simple user input with saving the key into a file!

    argv = sys.argv
    if len(argv)<3:
        print_program_usage()
        sys.exit(-1)

    if len(argv)==3:
        seed_num_str = argv[1]
        path_to_save_str = argv[2]

        try:
           seed_num = int(seed_num_str)
        except ValueError:
            seed_num = None

        if seed_num==None:
            print('2nd argument must be an integer number!')
            print_program_usage()
            sys.exit(-2)
    if len(argv)==4:
        if argv[1]=='-h':
            seed_num_str = argv[2]
            path_to_save_str = argv[3]
            try:
                seed_num = int(seed_num_str, 16)
            except ValueError:
                seed_num = None

            if seed_num==None:
                print('3nd argument must be an hex number!')
                print_program_usage()
                sys.exit(-3)
        else:
            print('Some wrong inputs!')
            print_program_usage()
            sys.exit(-4)

    bits = 16
    n = 2**bits
    sbox_ident = np.arange(0, n).astype(np.uint16)
    sbox0 = np.roll(sbox_ident, 1)
    len_sbox = sbox0.shape[0]
    print("sbox0:\n{}".format(sbox0))

    sbox1 = sbox0.copy()
    sbox2 = sbox0.copy()

    master_sbox = create_master_sbox_2byte(seed_num)

    print("master_sbox: {}".format(master_sbox))

    for i in range(0, 100):
        if i%10==0:
            print("i: {}".format(i))
            
        sbox1_new1 = np.argsort((np.cumsum(sbox2)+sbox1[sbox2])%n)
        sbox2_new1 = np.argsort((np.cumsum(sbox2)+sbox2[sbox1]+sbox1_new1)%n)

        sbox1_new2 = np.argsort((sbox1_new1+np.roll(sbox2_new1, 1))%n)
        sbox2_new2 = np.argsort((sbox2_new1+np.roll(sbox1_new2, 1)+np.roll(sbox2_new1, 1))%n)

        sbox1_new3 = master_sbox[sbox1_new2]

        sbox1 = restore_sbox(sbox1_new3, sbox2)
        sbox2 = restore_sbox(sbox2_new2, sbox1)

    if not '/' in path_to_save_str:
        path_dir = './'
        file_name = path_to_save_str
    else:
        l_split = path_to_save_str.split('/')
        path_dir = '/'.join(l_split[:-1])+'/'
        file_name = l_split[-1]

    if not '.' in file_name:
        new_file_name = file_name
        extension = 'hex'
    else:
        l_split_name = file_name.split('.')
        new_file_name = '.'.join(l_split_name[:-1])
        extension = l_split_name[-1]

    master_sbox.astype(np.uint16).tofile(open(path_dir+new_file_name+'_master_key.{}'.format(extension), 'wb'))
    sbox1.astype(np.uint16).tofile(open(path_dir+new_file_name+'_sbox1.{}'.format(extension), 'wb'))
    sbox2.astype(np.uint16).tofile(open(path_dir+new_file_name+'_sbox2.{}'.format(extension), 'wb'))

        # print("- sbox1: {}".format(sbox1))
        # print("- sbox2: {}".format(sbox2))

    # l_sbox_tpl = [tuple(sbox0.tolist())]
    # s_sbox_tpl = set(l_sbox_tpl)
    # for i in range(0, 10):
        
        # # del sbox1_prev
        # sbox1_prev = sbox1.copy()
        # sbox1 = np.argsort((np.cumsum(sbox1)+sbox1_prev)%len_sbox)
        # print("i: {}, sbox1:\n{}".format(i, sbox1))
        # idxs = sbox1==sbox_ident
        # idxs_ident = np.where(idxs)[0]
        # if idxs_ident.shape[0]>0:
        #     print("- idxs_ident: {}, idxs_ident.shape[0]: {}".format(idxs_ident, idxs_ident.shape[0]))
        #     # print("i: {}, idxs_ident: {}, idxs_ident.shape[0]: {}".format(i, idxs_ident, idxs_ident.shape[0]))
        #     if idxs_ident.shape[0]==1:
        #         # print('Only 1 ident position found!')
        #         idx = idxs_ident[0]
        #         for j in range(0, len_sbox):
        #             k = sbox1_prev[j]
        #             if k!=idx:
        #                 # change with this position!
        #                 sbox1[j], sbox1[k] = sbox1[k], sbox1[j]
        #     else:
        #         # TODO: change instead of normal rolling (which will always work!)
        #         # take the argsort argument from the splice of the sbox1 array! if nothing found, do the roll(arr, 1)
        #         len_idxs_ident = idxs_ident.shape[0]
        #         idxs_ident_ident = np.arange(0, len_idxs_ident)
        #         is_found_roll = False
        #         for j in range(0, len_sbox-len_idxs_ident+1):
        #             idxs_roll = np.argsort(sbox1_prev[j:j+len_idxs_ident])
        #             if np.any(idxs_roll==idxs_ident_ident):
        #                 continue
        #             is_found_roll = True
        #             break

        #         if is_found_roll:
        #             sbox1[idxs_ident] = sbox1[idxs_ident[idxs_roll]]
        #             del idxs_roll
        #             continue

        #         sbox1_prev_part = np.hstack((sbox1_prev[-len_idxs_ident+1:], sbox1_prev[:len_idxs_ident-1]))
        #         for j in range(0, len_idxs_ident):
        #             idxs_roll = np.argsort(sbox1_prev[j:j+len_idxs_ident])
        #             if np.any(idxs_roll==idxs_ident_ident):
        #                 continue
        #             is_found_roll = True
        #             break

        #         if is_found_roll:
        #             sbox1[idxs_ident] = sbox1[idxs_ident[idxs_roll]]
        #             del idxs_roll
        #             continue

        #         sbox1[idxs_ident] = sbox1[np.roll(idxs_ident, 1)]

        # t = tuple(sbox1.tolist())
        # if t in s_sbox_tpl:
        #     l_sbox_tpl.append(t)
        #     break
        # l_sbox_tpl.append(t)
        # s_sbox_tpl.add(t)

    # l_full_cycle_sbox = l_sbox_tpl[l_sbox_tpl.index(l_sbox_tpl[-1]):-1]
    # print("l_full_cycle_sbox: {}".format(l_full_cycle_sbox))
    # print("n: {}, len(l_full_cycle_sbox): {}".format(n, len(l_full_cycle_sbox)))
