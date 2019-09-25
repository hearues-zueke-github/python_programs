#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time
from functools import reduce

from collections import defaultdict

from PIL import Image

import numpy as np

sys.path.append("..")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

import decimal
from decimal import Decimal as Dec
decimal.getcontext().prec = 1000


def time_measure(f, args):
    start_time = time()
    ret = f(*args)
    end_time = time()
    diff_time = end_time-start_time
    return ret, diff_time


class ShowImg(Frame, object):
    def __init__(self, img):
        parent = Tk()
        Frame.__init__(self, parent)
        self.pack(fill=BOTH, expand=1)
        label1 = Label(self)
        label1.photo= ImageTk.PhotoImage(img)
        label1.config(image=label1.photo)
        label1.pack(fill=BOTH, expand=1)
        parent.mainloop()


def convert_1d_to_2d_arr(arr, length):
    arr_2d = np.zeros((arr.shape[0]-length+1, length), dtype=np.uint8)
    for i in range(0, length-1):
        arr_2d[:, i] = arr[i:-length+1+i]
    arr_2d[:, -1] = arr[length-1:]
    return arr_2d


lst_int_base_82 = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_.,:;!?#$%&()[]{}/\\ ")
base_82_len = len(lst_int_base_82)
dict_base_82_int = {v: i for i, v in enumerate(lst_int_base_82, 0)}
# first convert secret to an int and then to a binary number!
# secret = "abcde"

lst_prefix = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
lst_suffix = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]

jumps_amount = 200

# calc the probability for getting a random lst_bits which is equal to a secret lst_bits list!
# e.g. lst_bits is 100 bits long, so 16 for prefix, 16 for suffix, ca 61 for secret and 7 for length!
# P(2**100) = 1/2**100 -> which is very very tinny!

# e.g. image size is 2**12*2**12 = 2**24 bits
# find naive way the same bits -> 2**24 possibilities!
# suppose reverse bits, invert bits, both, rotate bits etc.
# would be: 2*2*100=400 ~ 2**9 possibilities
# reading the bits from image per rows or per cols:
# from top to bottom each row from left to right
# from top to bottom each row from right to left
# etc...
# 8 different ways possible ~ 2**3
# reading rows or cols alternating ways -> 8 more ways possible: 16 in total
# reading antidiagonal -> again 16 ways: total 32, which is ~2**%
# product of all ways of combination for an 2**12*2**12 image -> 2**24 * 2**9 * 2**5
# -> 2**38 -> still less than 2**100!!! The probability that the secret_concent would be
# found at least once in a very image is ~1/2**62!!
# the true exponent n for 1/2**n is calculated as follows:
# len(bin(int(1/(1-(1-Dec(1)/Dec(2**100))**(2**38))))[2:])
# But the probability is very very small (~1/10**18) that the secret could be found by random only!

def convert_base_82_to_int(num_base_82):
    b = 1
    s = 0
    for i, v in enumerate(reversed(list(num_base_82)), 0):
        n = dict_base_82_int[v]
        s += n*b
        b *= base_82_len
    return s


def convert_int_to_base_82(num_int):
    l = []
    while num_int > 0:
        l.append(num_int % base_82_len)
        num_int //= base_82_len
    n = list(map(lambda x: lst_int_base_82[x], reversed(l)))
    return "".join(n)


def convert_int_to_lst_bin(n):
    return list(map(int, bin(n)[2:]))


def convert_lst_bin_to_int(l):
    arr = np.array(l, dtype=object)
    length = arr.shape[0]
    return np.sum(arr*2**np.arange(length-1, -1, -1).astype(object))


def add_secret_to_pix(pix, secret, jumps, offset, invalid_idxs=np.array([])):
    """ TODO: add description!
    """

    s = convert_base_82_to_int(secret)
    s_lst_bin = list(map(int, bin(s)[2:]))
    # current secret bits: prefix|secret_bits|length|suffix
    # # secret bits: prefix|secret_bits|length|checksum|suffix
    s_lst = lst_prefix+s_lst_bin+convert_int_to_lst_bin(len(s_lst_bin))+lst_suffix
    s_arr = np.array(s_lst)
    length = s_arr.shape[0]

    pix = pix.copy()

    if len(pix.shape) == 2:
        h, w = pix.shape
        c = 1
    else:
        h, w, c = pix.shape

    pix = pix.reshape((-1, c))

    red_1_bit = pix[:, 0]&0x1
    green_1_bit = pix[:, 1]&0x1
    blue_1_bit = pix[:, 2]&0x1

    # find the bits, which need to be changed!
    idxs = np.arange(0, length)*jumps+offset
    arr_xor_bit = red_1_bit^green_1_bit^blue_1_bit
    arr_xor_bit_idxs = arr_xor_bit[idxs]

    idxs_diff = np.where(arr_xor_bit_idxs!=s_arr)[0]

    idxs_need_change = idxs[idxs_diff]

    print("s_arr.shape: {}".format(s_arr.shape))
    print("idxs_diff.shape: {}".format(idxs_diff.shape))

    idxs_c = np.random.randint(0, c, (idxs_need_change.shape[0], ))
    vals = pix[idxs_need_change, idxs_c]
    pix[idxs_need_change, idxs_c] = (vals&0xFE)|((vals&0x1)^1)

    pix = pix.reshape((h, w, c))

    return pix


def find_possible_secrets(arr):
    """ Find possible secrets which are contained in arr!

    Keyword arguments:
    arr -- numpy array with 0's or 1's
    """

    assert isinstance(arr, np.ndarray)
    assert np.all((arr==0)|(arr==1))
    assert len(arr.shape)==1

    bit_field_length = len(lst_prefix)
    bit_field_prefix = np.array(lst_prefix)
    bit_field_suffix = np.array(lst_suffix)

    found_secrets = []

    length = arr.shape[0]
    for jumps in range(1, jumps_amount+1):
        for offset in range(0, jumps):
            # TODO: find indexes for [0, 0, 1, 1, 0, 0, 1, 1] and
            # [0, 1, 1, 0, 1, 0, 0, 1] for getting the secrets!!!
            idxs = np.arange(offset, length, jumps)
            arr_1d = arr[idxs]
            arr_2d = convert_1d_to_2d_arr(arr_1d, bit_field_length)
            idxs1 = np.all(arr_2d==bit_field_prefix, axis=1)
            idxs2 = np.all(arr_2d==bit_field_suffix, axis=1)

            idxs1_pos = np.where(idxs1)[0]
            idxs2_pos = np.where(idxs2)[0]

            if len(idxs1_pos)==0 or len(idxs2_pos)==0:
                continue

            for idx1 in idxs1_pos:
                for idx2 in idxs2_pos:
                    if idx1+bit_field_length>=idx2:
                        continue
                    
                    content = arr[idxs[idx1:idx2+bit_field_length]]
                    secret_content = content[bit_field_length:-bit_field_length]
                    length_secret_content = secret_content.shape[0]
                    # TODO: find the correct length of the secret array!
                    for i in range(1, len(secret_content)):
                        length_secret = convert_lst_bin_to_int(secret_content[-i:])
                        if length_secret==length_secret_content-i:
                            secret_arr = secret_content[:-i]
                            secret = convert_int_to_base_82(convert_lst_bin_to_int(secret_arr))
                            found_secrets.append(secret)
                        elif length_secret>length_secret_content-i:
                            break

    return found_secrets


secret_test = "test123%$&/?!-_,:.;"
assert secret_test==convert_int_to_base_82(convert_base_82_to_int(secret_test))
assert 12345678901234567890==convert_base_82_to_int(convert_int_to_base_82(12345678901234567890))

assert 1234567==convert_lst_bin_to_int(convert_int_to_lst_bin(1234567))
assert [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]==convert_int_to_lst_bin(convert_lst_bin_to_int([1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]))

if __name__ == "__main__":
    print("Hello World!")
    
    img_src_path = "images/orig_image_3_no_secret.jpg"
    img_dst_path = "images/orig_image_3_with_secret.jpg"
    # img_src_path = "images/orig_image_2_no_secret.png"
    # img_dst_path = "images/orig_image_2_with_secret.png"

    img = Image.open(img_src_path)
    pix = np.array(img)
    if len(pix.shape)==3:
        if pix.shape[2]==4:
            pix = pix[..., :3]
            Image.fromarray(pix).save(img_src_path)

    img = Image.open(img_src_path)
    # img.show()
    pix_orig = np.array(img)

    pix = pix_orig.copy()

    secret = "Test123!!!?!$&%"
    jumps = 145
    offset = 40000
    pix = add_secret_to_pix(pix, secret, jumps, offset)

    secret = "My_First_Real_secret_MESSAGE!?.:,;"
    jumps = 4
    offset = 2000
    pix = add_secret_to_pix(pix, secret, jumps, offset)

    secret = "123_One_more_message!_Why_not?!"
    jumps = 8
    offset = 100000
    pix = add_secret_to_pix(pix, secret, jumps, offset)

    img2 = Image.fromarray(pix)
    print("Save image '{}'!".format(img_dst_path))
    img2.save(img_dst_path, quality=100, subsampling=0)

    # img2 = Image.fromarray(((pix^pix_orig)*255).astype(np.uint8))
    # img2 = Image.fromarray(pix)
    # img2.save(img_dst_path)
    # img2.show()

    print("Load image '{}'!".format(img_dst_path))
    img3 = Image.open(img_dst_path)
    pix3 = np.array(img3)

    pix_1d = pix3.reshape((-1, pix3.shape[-1]))
    red_1_bit = pix_1d[:, 0]&0x1
    green_1_bit = pix_1d[:, 1]&0x1
    blue_1_bit = pix_1d[:, 2]&0x1
    s_arr_restore = red_1_bit^green_1_bit^blue_1_bit

    print("Find possible secrets in image '{}'!")
    lst_possible_secrets = find_possible_secrets(s_arr_restore)
    print("lst_possible_secrets: {}".format(lst_possible_secrets))

    Image.fromarray((pix3^pix_orig)*255).show()
