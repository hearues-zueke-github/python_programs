#! /usr/bin/python3

# -*- coding: utf-8 -*-

import argparse
import dill
import hashlib
import marshal
import pickle
import os
import sys

import multiprocessing as mp
import numpy as np
import pandas as pd

from dotmap import DotMap
from multiprocessing import Process, Queue

from time import time

import Utils

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

SIGNATURE_ARRAY = np.array([0x81, 0x89, 0x96, 0x91, 0x53, 0x14, 0x15, 0x92, 0x65, 0x35], dtype=np.uint8)

class SBox(Exception):
    def __init__(self, sbox, bits=8):
        assert bits==8 or bits==16
        self.sbox = np.array(sbox)
        self.bits = bits
        self.length = self.sbox.shape[0]
        if bits==8:
            self.check_sbox_8bit()
        elif bits==16:
            self.check_sbox_16bit()


    def check_sbox_8bit(self):
        sbox = self.sbox
        if sbox.shape[0]!=256:
            self.sbox_ok = False
            return

        if np.any(sbox==np.arange(0, 256)):
            self.sbox_ok = False
            return

        if np.unique(sbox).shape[0]!=256:
            self.sbox_ok = False
            return

        self.sbox_ok = True


    def check_sbox_16bit(self):
        sbox = self.sbox
        if sbox.shape[0]!=65536:
            self.sbox_ok = False
            return

        if np.any(sbox==np.arange(0, 65536)):
            self.sbox_ok = False
            return

        if np.unique(sbox).shape[0]!=65536:
            self.sbox_ok = False
            return

        self.sbox_ok = True


class SBoxes(Exception):
    def __init__(self, seed, sbox_amount, bits=8):
        self.seed = seed
        self.sbox_amount = sbox_amount
        self.bits = bits

        self.arr_sboxes_obj = self.calculate_sboxes(seed, sbox_amount, bits)
        self.arr_sboxes = np.array([sbox_obj.sbox for sbox_obj in self.arr_sboxes_obj])


    def gen_count_array(self, n, m, max_arr=0, rounds=1):
        assert max_arr>0 or rounds>0
        r = 0
        c = 0
        arr = np.zeros((n, ), dtype=np.int)
        yield arr
        while True:
            for i in range(0, n):
                arr[i] += 1
                if arr[i]>=m:
                    arr[i] = 0
                    if rounds>0 and i==n-1:
                        r += 1
                        if r>=rounds:
                            return
                else:
                    break

            c += 1
            if max_arr>0 and c>=max_arr:
                return

            yield arr.copy()


    def get_simple_sbox_16(self):
        sbox_16 = np.arange(0, 2**16).astype(np.uint16)
        sbox_16 = np.roll(sbox_16, 1)
        # print("sbox_16: {}".format(sbox_16))

        i = 0
        for j1 in range(0, 65536):
            i = (i+377)%65535

            j1 = j1%65536
            j2 = (j1+i)%65536
            
            v1 = sbox_16[j1]
            v2 = sbox_16[j2]

            if j1!=v2 and j2!=v1:
                sbox_16[j1] = v2
                sbox_16[j2] = v1

        return sbox_16


    def simple_16bit_encryption(self, arr, sbox_16):
        assert arr.dtype==np.uint8
        arr = arr.copy()
        
        for r in range(0, 2): # rounds!
            n = arr.shape[0]
            for i in range(0, n):
                i2 = (i+1)%n
                
                b1 = arr[i]
                b2 = arr[i2]
                b = b1<<8|b2

                b_new = sbox_16[b]
                b1_new = b_new>>8
                b2_new = b_new&0xFF

                arr[i] = b1_new
                arr[i2] = b2_new

        return arr


    def get_pseudo_random_8bit_array(self, n, size):
        assert n>=0
        lb = bin(n)[2:]
        s = len(lb)
        l = np.array(list(map(int, lb)))
        size_bits = size*8
        i = 1
        while True:
            lnew = np.roll(np.hstack((l, np.roll(l, i))), i)
            l = np.bitwise_xor.reduce((lnew, np.roll(np.hstack((l, (l+1)%2)), -i)), axis=0)
            s *= 2
            if s>=size_bits:
                break
            i += 1

        arr = l[:l.shape[0]-(l.shape[0]%8)]
        arr = arr[:size*8].reshape((-1, 8))
        arr_num = np.sum(arr*2**np.arange(7, -1, -1), axis=1).astype(np.uint8)

        return arr_num


    def calculate_sboxes_16_bits(self, seed, sbox_amount):
        mod_num = 2**16

        arr_str = np.array(list((lambda s: s+'0'*(3-(len(s)-1)%4))(hex(seed)[2:])), dtype='<U4')
        arr_str = arr_str.reshape((-1, 4)).T
        charadd = np.core.defchararray.add
        arr_hex_str = (lambda arr: charadd("0x", charadd(arr[0], charadd(arr[1], charadd(arr[2], arr[3])))))(arr_str)
        arr = np.vectorize(lambda x: int(x, 16))(arr_hex_str).astype(np.uint16)
        
        if arr.shape[0]%(mod_num*sbox_amount)!=0:
            arr = np.hstack((arr, np.zeros((mod_num*sbox_amount-(arr.shape[0]%(mod_num*sbox_amount)), ), dtype=np.uint16)))
        arr_seed = np.bitwise_xor.reduce(arr.reshape((-1, mod_num*sbox_amount)), axis=0).astype(np.uint16)
        
        arr_rand = self.get_pseudo_random_8bit_array(sbox_amount, mod_num*sbox_amount*2)
        arr_rand = arr_rand.view(np.uint16)

        def get_zero_sbox_byte_counter():
            return np.zeros((65536, ), dtype=np.int)
        def get_zero_arr_sboxes_lst():
            return np.frompyfunc(list, 0, 1)(np.empty((sbox_amount, ), dtype=object)) 

        sbox_16 = self.get_simple_sbox_16()

        num_iter = 0
        block_size = 32
        assert arr_seed.shape[0]%block_size==0
        mod_block = arr_seed.shape[0]//block_size
        lst_sboxes_ok = []
        arr_iter = arr_seed.copy()
        sbox_byte_counter = get_zero_sbox_byte_counter()
        arr_sboxes_lst = get_zero_arr_sboxes_lst()
        while len(lst_sboxes_ok)<sbox_amount:
            # print("num_iter: {}".format(num_iter))
            it_block = num_iter%mod_block
            arr_iter_part = arr_iter[block_size*it_block:block_size*(it_block+1)]
            arr_iter_part_encrypt = self.simple_16bit_encryption(arr_iter_part.view(np.uint8), sbox_16).view(np.uint16)
            arr_iter_encrypt = arr_iter.reshape((-1, block_size))^arr_iter_part_encrypt
            arr_iter = np.roll(arr_iter_encrypt.flatten(), 1)
            arr_iter = (arr_iter^arr_rand)

            for i in arr_iter:
                if sbox_byte_counter[i]<sbox_amount:
                    arr_sboxes_lst[sbox_byte_counter[i]].append(i)
                    sbox_byte_counter[i] += 1

            lens = np.frompyfunc(len, 1, 1)(arr_sboxes_lst)
            print("- lens: {}".format(lens))
            idxs = lens==mod_num
            if np.any(idxs):
                print('At least one bin full!')
                arr_sboxes_obj = np.array([SBox(sbox=np.roll(np.array(sbox), j), bits=16) for j, sbox in enumerate(arr_sboxes_lst[idxs], 1)])
                lst_ok = [sbox.sbox_ok for sbox in arr_sboxes_obj]

                print("np.sum(lst_ok): {}".format(np.sum(lst_ok)))
                if np.any(lst_ok):
                    lst_sboxes_ok.extend(arr_sboxes_obj[lst_ok])

                    if len(lst_sboxes_ok)>=sbox_amount:
                        lst_sboxes_ok = lst_sboxes_ok[:sbox_amount]
                        break

                sbox_byte_counter = get_zero_sbox_byte_counter()
                arr_sboxes_lst = get_zero_arr_sboxes_lst()

            num_iter += 1

        return np.array(lst_sboxes_ok)


    def calculate_sboxes_8_bits(self, seed, sbox_amount):
        mod_num = 2**8

        arr_str = np.array(list((lambda s: s[2:(lambda n: n-n%2)(len(s))])(hex(seed)))).reshape((-1, 2)).T
        arr_hex_str = (lambda arr: np.core.defchararray.add("0x", np.core.defchararray.add(arr[0], arr[1])))(arr_str)
        arr = np.vectorize(lambda x: int(x, 16))(arr_hex_str).astype(np.uint8)
        
        if arr.shape[0]%(mod_num*sbox_amount)!=0:
            arr = np.hstack((arr, np.zeros((mod_num*sbox_amount-(arr.shape[0]%(mod_num*sbox_amount)), ), dtype=np.uint8)))
        
        arr = np.bitwise_xor.reduce(arr.reshape((-1, mod_num*sbox_amount)), axis=0)
        arr_rand = self.get_pseudo_random_8bit_array(sbox_amount, mod_num*sbox_amount)

        def get_zero_sbox_byte_counter():
            return np.zeros((256, ), dtype=np.int)
        def get_zero_arr_sboxes_lst():
            return np.frompyfunc(list, 0, 1)(np.empty((sbox_amount, ), dtype=object)) 

        num_iter = 0
        lst_sboxes_ok = []
        sbox_byte_counter = get_zero_sbox_byte_counter()
        arr_sboxes_lst = get_zero_arr_sboxes_lst()
        while len(lst_sboxes_ok)<sbox_amount:
            print("num_iter: {}".format(num_iter))
            arr_used = arr^np.roll(arr_rand^(num_iter%mod_num), num_iter)

            for i in arr_used:
                if sbox_byte_counter[i]<sbox_amount:
                    arr_sboxes_lst[sbox_byte_counter[i]].append(i)
                    sbox_byte_counter[i] += 1

            idxs = np.frompyfunc(len, 1, 1)(arr_sboxes_lst)==mod_num
            if np.any(idxs):
                print('At least one bin full!')
                arr_sboxes_obj = np.array([SBox(sbox=np.roll(np.array(sbox), j), bits=8) for j, sbox in enumerate(arr_sboxes_lst[idxs], 1)])
                lst_ok = [sbox.sbox_ok for sbox in arr_sboxes_obj]

                if np.any(lst_ok):
                    lst_sboxes_ok.extend(arr_sboxes_obj[lst_ok])

                    if len(lst_sboxes_ok)>=sbox_amount:
                        lst_sboxes_ok = lst_sboxes_ok[:sbox_amount]
                        break

                sbox_byte_counter = get_zero_sbox_byte_counter()
                arr_sboxes_lst = get_zero_arr_sboxes_lst()

            num_iter += 1

        return np.array(lst_sboxes_ok)


    def calculate_sboxes(self, seed, sbox_amount, bits=8):
        assert bits==8 or bits==16

        if bits==8:
            return self.calculate_sboxes_8_bits(seed, sbox_amount)
        if bits==16:
            return self.calculate_sboxes_16_bits(seed, sbox_amount)


def convert_str_hex_to_arr_uint8(str_hex):
    assert isinstance(str_hex, str) and len(str_hex)%2==0
    return np.array([int(str_hex[2*i:2*(i+1)], 16) for i in range(0, len(str_hex)//2)], dtype=np.uint8)


def apply_encrypt_16bits(plain_8bit, arr_sboxes_16, block_size_8bit=8, rounds=4):
    assert block_size_8bit%2==0
    arr = np.array(plain_8bit, dtype=np.uint8)
    orig_size = arr.shape[0]
    print("orig_size: {}".format(orig_size))
    arr_size = np.array([orig_size], dtype=np.int64).view(np.uint8)
    if orig_size%block_size_8bit!=0:
        arr = np.hstack((arr, np.zeros((block_size_8bit-orig_size%block_size_8bit, ), dtype=np.uint8)))

    mask = np.arange(0, block_size_8bit, dtype=np.uint8)
    arr = arr.reshape((-1, block_size_8bit))
    i_sbox1 = 0
    i_sbox2 = 1
    for _ in range(0, rounds):
        for r in arr:
            sbox1 = arr_sboxes_16[i_sbox1]
            sbox2 = arr_sboxes_16[i_sbox2]

            mask = sbox2[np.roll(sbox1[mask.view(np.uint16)].view(np.uint8), 1).view(np.uint16)].view(np.uint8)
            
            r[:] = r^mask
            r[:] = sbox2[np.roll(sbox1[r.view(np.uint16)].view(np.uint8), 1).view(np.uint16)].view(np.uint8)

            i_sbox1 = (i_sbox1+1)%arr_sboxes_16.shape[0]
            i_sbox2 = (i_sbox2+1)%arr_sboxes_16.shape[0]

    readable_hash_plain = hashlib.sha256(plain_8bit).hexdigest()
    arr_sha256_plain = convert_str_hex_to_arr_uint8(readable_hash_plain)

    arr_encrypt = np.hstack((arr_sha256_plain, arr_size, arr.flatten()))    

    readable_hash_encrypt = hashlib.sha256(arr_encrypt).hexdigest()
    arr_sha256_encrypt = convert_str_hex_to_arr_uint8(readable_hash_encrypt)

    return np.hstack((SIGNATURE_ARRAY, arr_sha256_encrypt, arr_encrypt))


def apply_decrypt_16bits(cypher_8bit, arr_sboxes_16, arr_sboxes_inv_16, block_size_8bit=8, rounds=4):
    assert block_size_8bit%2==0
    assert arr_sboxes_16.shape[0]==arr_sboxes_inv_16.shape[0]
    
    sbox_amount = arr_sboxes_inv_16.shape[0]

    arr = np.array(cypher_8bit, dtype=np.uint8)
    using_bytes = [0, SIGNATURE_ARRAY.shape[0], 32, 32, 8] # signature length, encrypt hash sha256, plain hash sha256, size of orig file
    using_bytes_pos = np.cumsum(using_bytes)
    arr_sig = arr[using_bytes_pos[0]:using_bytes_pos[1]]
    assert np.all(arr_sig==SIGNATURE_ARRAY)
    arr_sha256_encrypt = arr[using_bytes_pos[1]:using_bytes_pos[2]]

    readable_hash_encrypt = hashlib.sha256(arr[using_bytes_pos[2]:]).hexdigest()
    arr_sha256_encrypt_calc = convert_str_hex_to_arr_uint8(readable_hash_encrypt)

    assert np.all(arr_sha256_encrypt==arr_sha256_encrypt_calc)

    arr_sha256_plain = arr[using_bytes_pos[2]:using_bytes_pos[3]]
    arr_size = arr[using_bytes_pos[3]:using_bytes_pos[4]]
    orig_size = arr_size.view(np.int64)[0]


    mask = np.arange(0, block_size_8bit, dtype=np.uint8)
    arr = arr[using_bytes_pos[4]:].reshape((-1, block_size_8bit))
    arr = np.flip(arr, axis=0)
    i_sbox = 0
    i_sbox2 = 1
    for _ in range(0, rounds):
        for _ in range(0, arr.shape[0]):
            sbox1 = arr_sboxes_16[i_sbox]
            sbox2 = arr_sboxes_16[i_sbox2]

            mask = sbox2[np.roll(sbox1[mask.view(np.uint16)].view(np.uint8), 1).view(np.uint16)].view(np.uint8)

            i_sbox = (i_sbox+1)%sbox_amount
            i_sbox2 = (i_sbox2+1)%sbox_amount

    for _ in range(0, rounds):
        for r in arr:
            i_sbox = (i_sbox-1)%sbox_amount
            i_sbox2 = (i_sbox2-1)%sbox_amount
            sbox_inv1 = arr_sboxes_inv_16[i_sbox]
            sbox_inv2 = arr_sboxes_inv_16[i_sbox2]

            r[:] = sbox_inv1[np.roll(sbox_inv2[r.view(np.uint16)].view(np.uint8), -1).view(np.uint16)].view(np.uint8)
            r[:] = r^mask
            
            mask = sbox_inv1[np.roll(sbox_inv2[mask.view(np.uint16)].view(np.uint8), -1).view(np.uint16)].view(np.uint8)

    arr = np.flip(arr, axis=0).flatten()[:orig_size]

    readable_hash_plain = hashlib.sha256(arr).hexdigest()
    arr_sha256_plain_calc = convert_str_hex_to_arr_uint8(readable_hash_plain)

    assert np.all(arr_sha256_plain==arr_sha256_plain_calc)

    return arr


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-s', '--seed', metavar='seed', dest='seed', type=int, default=0,
                        help='A seed for the RNG (random number generator).')
    parser.add_argument('-n', '--sbox-amount', metavar='sbox-amount', dest='sbox_amount', type=int, default=1,
                        help='The amount of different sboxes.')
    parser.add_argument('-b', '--bits', metavar='bits', dest='bits', type=int, default=16,
                        help='Possible bits: 8 or 16.', choices=[8, 16])
    parser.add_argument('-p', '--path', metavar='path', dest='path', type=str, default='sboxes.hex',
                        help='Name of the file where to save it.', required=False)

    args = parser.parse_args()

    seed = args.seed
    sbox_amount = args.sbox_amount
    bits = args.bits

    path = args.path

    sboxes = SBoxes(seed=seed, sbox_amount=sbox_amount, bits=bits)

    s = sboxes.arr_sboxes
    assert np.all(np.array([np.sum(np.any(s!=si, axis=1))==s.shape[0]-1 for si in s]))
    
    with open(path, 'wb') as f:
        dill.dump({'seed': seed, 'sbox_amount': sbox_amount, 'bits': bits, 'arr_sboxes': s}, f)

    sys.exit(0)

    

    # read the file in as numbers in an array
    file_path = PATH_ROOT_DIR+'test_file_plain.txt'
    # file_path = '/bin/cat'
    with open(file_path, 'rb') as f:
        content = f.read()
    
    arr_plain = np.array(list(content), dtype=np.uint8)
    print("arr_plain: {}".format(arr_plain))

    arr_sboxes = sboxes.arr_sboxes
    arr_sboxes_inv = np.argsort(arr_sboxes, axis=1).astype(np.uint16)

    arr_encrypt = apply_encrypt_16bits(arr_plain, arr_sboxes, block_size_8bit=32, rounds=2)
    print("arr_encrypt: {}".format(arr_encrypt))

    arr_decrypt = apply_decrypt_16bits(arr_encrypt, arr_sboxes, arr_sboxes_inv, block_size_8bit=32, rounds=2)
    print("arr_decrypt: {}".format(arr_decrypt))

    assert np.all(arr_plain==arr_decrypt)

    with open('cat_plain', 'wb') as f:
        arr_plain.tofile(f)
    with open('cat_encrypt', 'wb') as f:
        arr_encrypt.tofile(f)
    with open('cat_decrypt', 'wb') as f:
        arr_decrypt.tofile(f)

    # sys.exit(0)

    # pd.DataFrame(sboxes.arr_sboxes).to_excel('sboxes.xlsx', index=False, header=False)
