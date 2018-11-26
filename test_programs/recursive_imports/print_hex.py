#! /usr/bin/python3.6

import os
import sys

import numpy as np

def print_arr_hexdump(arr):
    sys.stdout.write(" "*11)
    for i in range(0, 0x10):
        sys.stdout.write(" {:02X}".format(i))
    print("")

    amount = arr.shape[0]
    lines = amount // 0x10
    for row, line in enumerate(arr[:0x10*lines].reshape((-1, 0x10)), 1):
        sys.stdout.write("0x{:08X}:".format(row*0x10))

        for b in line:
            sys.stdout.write(" {:02X}".format(b))
        print("")

    if amount%0x10 > 0:
        row = lines+1
        sys.stdout.write("0x{:08X}:".format(row*0x10))

        for b in arr[lines*0x010:]:
            sys.stdout.write(" {:02X}".format(b))


if __name__ == "__main__":
    argv = sys.argv

    if len(argv) < 2:
        print("Need 2nd argument of file!")
        sys.exit(-1)


    file_path = argv[1]
    if not os.path.exists(file_path):
        print("Need a valid file path!")
        sys.exit(-2)


    with open(file_path, "rb") as fin:
        arr = np.fromfile(fin, dtype=np.uint8)

    print_arr_hexdump(arr)
    print("")
