#! /usr/bin/python3.5

import sys
# sys.path.append("../encryption")
sys.path.append("..")

import encryption.Utils as Utils

import numpy as np

from PIL import Image
def convert_num_to_hex_arr(num):
    return np.array(list(map(lambda x: int(x, 16), (lambda s: [s[i:i+2] for i in range(0, len(s), 2)])((lambda s: ("0" if len(s) % 2 == 1 else "")+s)(hex(num)[2:]))))).astype(np.uint8)

def convert_num_to_hex_arr_2(num):
    arr = []
    base = 2**64

    while num > 0:
        num, rest = divmod(num, base)
        arr.append(rest)
    
    # str_last = hex(num%base)[2:].lstrip("0")
    # last_length = (lambda x: x//2 if x%2==0 else x//2+1)(len(str_last))
    # print("str_last: {}".format(str_last))
    # print("last_length: {}".format(last_length))

    hex_arr = np.array(arr, dtype=np.uint64).view(np.uint8)[::-1]

    return hex_arr

def convert_hex_arr_to_num(hex_arr):
    return np.sum(hex_arr.astype(object)*256**np.arange(0, hex_arr.shape[0])[::-1].astype(object))

#num = 3**1292 # exp: min...1288, max...1292 for 256 bytes!
num = 7**8000000 # exp: min...1288, max...1292 for 256 bytes!
hex_arr = convert_num_to_hex_arr(num)
# hex_arr = convert_num_to_hex_arr_2(num)
# check_num = convert_hex_arr_to_num(hex_arr)

# print("num:\n{}".format(num))
# print("hex_arr:\n{}".format(hex_arr))
print("hex_arr:")
Utils.pretty_block_printer(hex_arr, 8, 0x200)
# Utils.pretty_block_printer(hex_arr, 8, hex_arr.shape[0])
# print("check_num:\n{}".format(check_num))

length = hex_arr.shape[0]
scale = int(np.sqrt(length//3//3//4))
pix = hex_arr[:3*scale*4*scale*3].astype(np.uint8).reshape((3*scale, 4*scale, 3))
print("pix.shape: {}".format(pix.shape))

img = Image.fromarray(pix)
img.show()
