#! /usr/bin/python2.7

import os
import sys

import numpy as np

np.set_printoptions(formatter={'int': lambda x: "{:02X}".format(x)})

def pretty_block_printer(block, bits, length):
    digits = int((bits+3)/4)
    temp = length-1
    digits_cols = 0
    while temp > 0:
        temp >>= 4
        digits_cols += 1

    line = ""
    line += " {}\x1b[1;31;38m".format(" "*digits_cols)
    for i in range(16):
        line += " {{:0{}X}}".format(digits).format(i)
    print(line)

    rows = length // 16
    for i in range(rows):
        line = ""
        line += "\x1b[1;35;38m{{:0{}X}}:\x1b[0m".format(digits_cols).format(i*16)
        for j in range(16):
            line += " {{:0{}X}}".format(digits).format(block[i*16+j])
        print(line)

    cols = length%16
    if cols != 0:
        i = rows
        line = ""
        line += "\x1b[1;35;38m{{:0{}X}}:\x1b[0m".format(digits_cols).format(i*16)
        for j in range(cols):
            line += " {{:0{}X}}".format(digits).format(block[i*16+j])
        print(line)

def encrypt_arr(arr, sbox):
    block_size = 16
    in_block_bytes = 8
    length_orig = arr.shape[0]
    block_amount = length_orig//in_block_bytes+((length_orig%in_block_bytes)!=0)
    print("length_orig: {:X}".format(length_orig))
    print("block_amount: {}".format(block_amount))

    # add the length_orig at the beginning of the arr, and add random bytes too
    # e.g. block size is 16

    length_arr = np.array([length_orig], dtype=np.uint64).view(np.uint8)
    rand_bytes = np.random.randint(0, 256, (8, )).astype(np.uint8)

    first_block = np.hstack((length_arr, rand_bytes))
    print("first_block:")
    pretty_block_printer(first_block, 8, block_size)

    arr_new = np.zeros((block_size*(block_amount+1), ), dtype=np.uint8)
    arr_new[:block_size] = encrypt_arr_block(first_block, sbox)
    for i in xrange(1, block_amount+1):
        if i == block_amount and length_orig % in_block_bytes != 0:
            block_arr = arr[in_block_bytes*(i-1):]
            block_arr = np.hstack((block_arr, np.random.randint(0, 256, (in_block_bytes-(length_orig%in_block_bytes))).astype(np.uint8)))
        else:
            block_arr = arr[in_block_bytes*(i-1):in_block_bytes*i]
        block_arr = np.hstack((block_arr, np.random.randint(0, 256, (block_size-len(block_arr), )).astype(np.uint8)))
        print("i: {}, block_arr:".format(i))
        pretty_block_printer(block_arr, 8, block_size)
        arr_new[block_size*i:block_size*(i+1)] = encrypt_arr_block(block_arr, sbox)

    return arr_new

def decrypt_arr(arr, sbox):
    block_size = 16
    in_block_bytes = 8
    length = arr.shape[0]
    assert len(arr)%block_size==0
    block_amount = length//block_size-1
    print("block_amount: {}".format(block_amount))
    # get the length_orig at the beginning of the arr, and add random bytes too
    # e.g. block size is 16

    length_arr = decrypt_arr_block(arr[:16], sbox)[:8]
    print("length_arr:")
    pretty_block_printer(length_arr, 8, 8)

    length_orig = int(length_arr.view(np.uint64)[0])
    arr_new = np.zeros((length_orig, ), dtype=np.uint8)

    for i in xrange(1, block_amount+1):
        block_arr = decrypt_arr_block(arr[block_size*i:block_size*(i+1)], sbox)
        if i == block_amount and length_orig%in_block_bytes!=0:
            arr_new[in_block_bytes*(i-1):] = block_arr[:length_orig%in_block_bytes]
        else:
            arr_new[in_block_bytes*(i-1):in_block_bytes*i] = block_arr[:in_block_bytes]
        print("i: {}, block_arr:".format(i))
        pretty_block_printer(block_arr, 8, len(block_arr))

    return arr_new

def encrypt_arr_block(arr, sbox):
    arr = arr+0
    length = len(arr)

    arr[0] = sbox[arr[0]]
    for i in xrange(1, length):
        arr[i] = sbox[arr[i] ^ arr[i-1]]

    arr[-1] = sbox[arr[-1]]
    for i in xrange(length-2, -1, -1):
        arr[i] = sbox[arr[i] ^ arr[i+1]]

    return arr

def decrypt_arr_block(arr, sbox):
    arr = arr+0
    length = len(arr)

    for i in xrange(0, length-1):
        arr[i] = sbox[arr[i]] ^ arr[i+1]
    arr[-1] = sbox[arr[-1]]

    for i in xrange(length-1, 0, -1):
        arr[i] = sbox[arr[i]] ^ arr[i-1]
    arr[0] = sbox[arr[0]]

    return arr

if __name__ == "__main__":
    f = lambda x: (8*x**2+3*x) % 256
    sbox_8bit = f(np.arange(0, 256))

    sbox_inv_8bit = sbox_8bit+0
    sbox_inv_8bit[sbox_8bit] = np.arange(0, 256)
    
    argv = sys.argv

    useage_text = "useage: ./crypt_file.py [en|de] <filename_in> <filename_out>"
    if len(argv) < 4:
        print(useage_text)
        sys.exit(-1)

    mode = argv[1]

    if mode != "en" and mode != "de":
        print("Wrong mode!")
        print(useage_text)
        sys.exit(-1)

    file_path_in = argv[2]
    file_path_out = argv[3]
    if not os.path.exists(file_path_in):
        print("File do not exists!")
        print(useage_text)

    with open(file_path_in, "rb") as fin:
        arr = np.array(bytearray(fin.read()), dtype=np.uint8)

    print("file in data:")
    pretty_block_printer(arr, 8, len(arr))

    if mode == "en":
        # encrypt
        arr_out = encrypt_arr(arr, sbox_8bit)
    else:
        # decrypt
        arr_out = decrypt_arr(arr, sbox_inv_8bit)

    print("file out data:")
    pretty_block_printer(arr_out, 8, len(arr_out))

    with open(file_path_out, "wb") as fout:
        fout.write(bytearray(arr_out))
