#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
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
            arr = np.hstack((arr, np.zeros((mod_num*sbox_amount-(arr.shape[0]%(mod_num*sbox_amount)), ), dtype=np.uint8)))
        
        arr_orig = np.bitwise_xor.reduce(arr.reshape((-1, mod_num*sbox_amount)), axis=0)
        arr_rand = self.get_pseudo_random_8bit_array(sbox_amount, mod_num*sbox_amount*2)
        arr_rand = arr_rand.view(np.uint16)

        def get_zero_sbox_byte_counter():
            return np.zeros((65536, ), dtype=np.int)
        def get_zero_arr_sboxes_lst():
            return np.frompyfunc(list, 0, 1)(np.empty((sbox_amount, ), dtype=object)) 

        num_iter = 0
        lst_sboxes_ok = []
        arr = arr_orig
        sbox_byte_counter = get_zero_sbox_byte_counter()
        arr_sboxes_lst = get_zero_arr_sboxes_lst()
        while len(lst_sboxes_ok)<sbox_amount:
            print("num_iter: {}".format(num_iter))
            arr = (arr^np.roll(arr_orig, 256)+1)%mod_num
            arr_used = arr^np.roll(arr_rand^(num_iter%mod_num), num_iter)

            for i in arr_used:
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
        
        mod_num = 2**bits

        arr_str = np.array(list((lambda s: s[2:(lambda n: n-n%2)(len(s))])(hex(seed)))).reshape((-1, 2)).T
        arr_hex_str = (lambda arr: np.core.defchararray.add("0x", np.core.defchararray.add(arr[0], arr[1])))(arr_str)
        arr = np.vectorize(lambda x: int(x, 16))(arr_hex_str).astype(np.uint8)
        if bits==16:
            arr = arr[:-1].astype(np.uint16)*256+arr[1:].astype(np.uint16)

        def get_zero_sbox_byte_counter():
            if bits==8:
                sbox_byte_counter = np.zeros((256, ), dtype=np.int)
            if bits==16:
                sbox_byte_counter = np.zeros((256*256, ), dtype=np.int)
            return sbox_byte_counter

        print("arr:\n{}".format(arr))
        # sbox_byte_counter = get_zero_sbox_byte_counter()

        num_iter = 0
        used_arr = arr
        lst_sboxes_ok = []
        l_rest_prev = []
        while len(lst_sboxes_ok)<sbox_amount:
            sbox_byte_counter = get_zero_sbox_byte_counter()
            arr_sboxes_lst = np.frompyfunc(list, 0, 1)(np.empty((sbox_amount, ), dtype=object)) 

            l_rest = []
            for i in used_arr:
                if sbox_byte_counter[i]<sbox_amount:
                    arr_sboxes_lst[sbox_byte_counter[i]].append(i)
                    sbox_byte_counter[i] += 1
                else:
                    l_rest.append(i)
            for i in l_rest_prev:
                if sbox_byte_counter[i]<sbox_amount:
                    arr_sboxes_lst[sbox_byte_counter[i]].append(i)
                    sbox_byte_counter[i] += 1
                else:
                    l_rest.append(i)

            l_rest_prev = l_rest
            print("len(l_rest): {}".format(len(l_rest)))

            # print("sbox_byte_counter: {}".format(sbox_byte_counter))
            
            # sbox_byte_counter = get_zero_sbox_byte_counter()
            # if not np.all(sbox_byte_counter==sbox_amount):
            #     num_iter += 1
            #     used_arr = (arr+num_iter)%mod_num
            #     continue

            # assert np.all(sbox_byte_counter==sbox_amount)



            # for idx, sbox in enumerate(arr_sboxes_lst):
            #     # print("idx: {}, len(sbox): {}".format(idx, len(sbox)))
            #     # Utils.pretty_block_printer(sbox, 8, len(sbox))
            #     # print("")
            #     continue

            arr_sboxes_obj = np.array([SBox(sbox=np.array(sbox), bits=bits) for sbox in arr_sboxes_lst if len(sbox)==mod_num])
            lst_ok = [sbox.sbox_ok for sbox in arr_sboxes_obj]
            # print("lst_ok: {}".format(lst_ok))

            lst_sboxes_ok.extend(arr_sboxes_obj[lst_ok])

            if len(lst_sboxes_ok)>=sbox_amount:
                lst_sboxes_ok = lst_sboxes_ok[:sbox_amount]
                break

            # lst_sboxes_ok = arr_sboxes_obj
            num_iter += 1
            used_arr = np.roll((arr+num_iter)%mod_num, num_iter)
        
        return np.array(lst_sboxes_ok)


def apply_encrypt(plain, arr_sboxes, block_size=8, rounds=4):
    arr = np.array(plain, dtype=np.uint8)
    orig_size = arr.shape[0]
    arr_size = np.array([orig_size], dtype=np.int64).view(np.uint8)
    if orig_size%block_size!=0:
        arr = np.hstack((arr, np.zeros((block_size-orig_size%block_size, ), dtype=np.uint8)))

    arr = arr.reshape((-1, block_size))
    i_sbox = 0
    for _ in range(0, rounds):
        for r in arr:
            sbox = arr_sboxes[i_sbox]

            for i in range(0, block_size):
                r[i] = sbox[(r[i]+i+i_sbox)%256]

            i_sbox = (i_sbox+1)%arr_sboxes.shape[0]

    return np.hstack((arr_size, arr.flatten()))

def apply_decrypt(cypher, arr_sboxes_inv, block_size=8, rounds=4):
    arr = np.array(cypher, dtype=np.uint8)
    arr_size = arr[:8]
    orig_size = arr_size.view(np.int64)[0]

    print("arr: {}".format(arr))
    arr = arr[8:].reshape((-1, block_size))
    arr = np.flip(arr, axis=0)
    i_sbox = (rounds*arr.shape[0])%arr_sboxes_inv.shape[0]
    for _ in range(0, rounds):
        for r in arr:
            i_sbox = (i_sbox-1)%arr_sboxes_inv.shape[0]
            
            sbox_inv = arr_sboxes_inv[i_sbox]

            for i in range(0, block_size):
                r[i] = (sbox_inv[r[i]]-i-i_sbox)%256

    arr = np.flip(arr, axis=0).flatten()[:orig_size]

    return arr


def gen_count_array(n, m, max_arr=0, rounds=1):
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


if __name__=="__main__":
    # # find a simple cycle for mod 255 in linear equation!
    # m = 2**16-1
    
    # for a in range(1, m):
    #     if a%100==0:
    #         print("a: {}".format(a))
    #     for c in range(377, 378):
    #         i = 0
    #         l = []
    #         for _ in range(0, m):
    #             i = (i*a+c)%m
    #             l.append(i)
    #         u, counts = np.unique(l, return_counts=True)
    #         # print("u.shape: {}".format(u.shape))

    #         try:
    #             assert u.shape[0]==m
    #             print("a: {}, c: {}".format(a, c))
    #         except:
    #             pass

    # sys.exit(0)

    # create a simple 16-bit sbox!
    def get_simple_sbox_16():
        sbox_16 = np.arange(0, 2**16).astype(np.uint16)
        sbox_16 = np.roll(sbox_16, 1)
        print("sbox_16: {}".format(sbox_16))

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

    # print("after: sbox_16: {}".format(sbox_16))

    def simple_16bit_encryption(arr, sbox_16):
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

    # arr = np.zeros((8, ), dtype=np.uint8)
    # arr[0] = 1

    # s = '['+', '.join(['0x{:02X}'.format(i) for i in arr])+']'
    # print("before:  arr: {}".format(s))

    # arr = simple_16bit_encryption(arr, sbox_16)

    # s = '['+', '.join(['0x{:02X}'.format(i) for i in arr])+']'
    # print("after:  arr: {}".format(s))

    # print("after:  arr: {}".format(arr))

    # # first try a 8-bit sbox!
    # sbox = np.arange(0, 2**8).astype(np.uint8)
    # sbox = np.roll(sbox, 1)
    # print("sbox: {}".format(sbox))

    # i = 0
    # for j1 in range(0, 256):
    #     i = (i*1+8)%255

    #     j1 = j1%256
    #     j2 = (j1+i)%256
        
    #     v1 = sbox[j1]
    #     v2 = sbox[j2]

    #     # print("v1: {}, v2: {}".format(v1, v2))
        
    #     if j1!=v2 and j2!=v1:
    #         sbox[j1] = v2
    #         sbox[j2] = v1

    #     # print("i: {}".format(i))
    # print("after: sbox: {}".format(sbox))


    # sys.exit(0)

    sbox_16 = get_simple_sbox_16()

    mod_num = 2**8
    arr_orig = np.array([0]*4, dtype=np.uint8)
    arr = arr_orig
    # arr_const = np.arange(0, arr_orig.shape[0]).astype(np.uint8)
    arr_acc = np.arange(0, arr_orig.shape[0]).astype(np.uint8)
    print("arr: {}".format(arr))
    
    l = [arr]

    m = 256
    print("m: {}".format(m))

    max_n = 500000
    # for i, arr_count in zip(range(1, max_n+1), gen_count_array(n=2, m=mod_num, max_arr=max_n, rounds=0)):
    # for i, arr_count in zip(range(1, max_n+1), gen_count_array(n=arr_orig.shape[0], m=mod_num, max_arr=max_n, rounds=0)):
    for i, arr_count in zip(range(1, max_n+1), gen_count_array(n=arr_orig.shape[0], m=m, max_arr=max_n, rounds=0)):
        if i%1000==0:
            print("i: {}".format(i))
        arr_count = arr_count.astype(np.uint8)
        arr = (arr^simple_16bit_encryption(arr_count, sbox_16))%mod_num
        # arr_acc = (((arr_acc^np.roll(arr_count, i))%mod_num)*5+3)%mod_num
        # arr = np.roll((((arr*13+3)%mod_num)^arr_acc)%mod_num, i)
        # print("i: {}".format(i))
        # print("- arr: {}".format(arr))
        l.append(arr)
    arrs = np.array(l, dtype=np.uint8)

    diffs = np.sum(np.abs(arrs-arrs[-1]), axis=1)
    idxs = np.where(diffs==0)[0]
    diffs2 = idxs[1:]-idxs[:-1]
    print("idxs: {}".format(idxs))
    print("diffs2: {}".format(diffs2))

    arrs_bytes_block = arrs.reshape((-1, )).view('u1'+',u1'*(arr_orig.shape[0]-1))
    u, c = np.unique(arrs_bytes_block, return_counts=True)

    sys.exit(0)

    seed = 16242345224354643249872943879857432095273454983759875987298
    sbox_amount = 32
    sboxes = SBoxes(seed=seed, sbox_amount=sbox_amount, bits=16)
    
    sys.exit(0)


    # these here is the key!
    seed = 3**280000+0
    sbox_amount = 2001
    # created sboxes
    sboxes = SBoxes(seed, sbox_amount, bits=8)

    # read the file in as numbers in an array
    file_path = '/bin/cat'
    with open(file_path, 'rb') as f:
        content = f.read()

    arr_plain = np.array(list(content), dtype=np.uint8)
    # arr_plain = np.array([0x00, 0x10, 0x20, 0x30, 0x40, 0x56, 0x98, 0xAB, 0xDE, 0xFF], dtype=np.uint8)
    print("arr_plain: {}".format(arr_plain))

    arr_sboxes = sboxes.arr_sboxes
    arr_sboxes_inv = np.argsort(arr_sboxes, axis=1)

    arr_encrypt = apply_encrypt(arr_plain, arr_sboxes, rounds=4)
    print("arr_encrypt: {}".format(arr_encrypt))
    
    arr_decrypt = apply_decrypt(arr_encrypt, arr_sboxes_inv, rounds=4)
    print("arr_decrypt: {}".format(arr_decrypt))

    assert np.all(arr_plain==arr_decrypt)

    with open('cat_plain', 'wb') as f:
        arr_plain.tofile(f)
    with open('cat_encrypt', 'wb') as f:
        arr_encrypt.tofile(f)
    with open('cat_decrypt', 'wb') as f:
        arr_decrypt.tofile(f)

    sys.exit(0)

    pd.DataFrame(sboxes.arr_sboxes).to_excel('sboxes.xlsx', index=False, header=False)
    s = sboxes.arr_sboxes
    assert np.all(np.array([np.sum(np.any(s!=si, axis=1))==s.shape[0]-1 for si in s]))


