#! /usr/bin/python2.7

import numpy as np

def encrypt_arr(arr, sbox):
    arr_new = arr+0
    length = len(s)

    arr_new[0] = sbox[arr_new[0]]
    for i in xrange(1, length):
        arr_new[i] = sbox[arr_new[i] ^ arr_new[i-1]]

    arr_new[-1] = sbox[arr_new[-1]]
    for i in xrange(length-2, -1, -1):
        arr_new[i] = sbox[arr_new[i] ^ arr_new[i+1]]

    return arr_new

def decrypt_arr(arr, sbox):
    arr_new = arr+0
    length = len(s)

    for i in xrange(0, length-1):
        arr_new[i] = sbox[arr_new[i]] ^ arr_new[i+1]
    arr_new[-1] = sbox[arr_new[-1]]

    for i in xrange(length-1, 0, -1):
        arr_new[i] = sbox[arr_new[i]] ^ arr_new[i-1]
    arr_new[0] = sbox[arr_new[0]]

    return arr_new

if __name__ == "__main__":
    s = "abcdef"

    f = lambda x: (8*x**2+3*x) % 256
    sbox_fix = f(np.arange(0, 256))
    # sbox_fix = np.vectorize(f)(np.arange(0, 256))
    # print("sbox_fix: {}".format(sbox_fix))
    sbox_fix_sort = np.sort(sbox_fix)
    # print("sbox_fix_sort: {}".format(sbox_fix_sort))

    # TODO: create a fix sbox, with a fixed formula
    sbox_8bit = f(np.arange(0, 256))
    # sbox_8bit = np.random.permutation(np.arange(0, 256)).astype(np.uint8)

    sbox_inv_8bit = sbox_8bit+0
    sbox_inv_8bit[sbox_8bit] = np.arange(0, 256)
    
    print("s: {}".format(s))

    # encrypt
    arr = np.array(bytearray(s)).astype(np.uint8)
    arr_en = encrypt_arr(arr, sbox_8bit)

    # decrypt
    arr_de = decrypt_arr(arr_en, sbox_inv_8bit)

    print("arr: {}".format(arr))
    print("arr_en: {}".format(arr_en))
    print("arr_de: {}".format(arr_de))

    print("str(bytearray(arr_de)): {}".format(str(bytearray(arr_de))))
