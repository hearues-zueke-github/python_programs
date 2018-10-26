#! /usr/bin/python3.6

import numpy as np

if __name__ == "__main__":
    a = np.random.randint(0, 256, (4, ), dtype=np.uint8)
    b = np.random.randint(0, 256, (4, ), dtype=np.uint8)

    get_arr_in_bits = lambda arr: "0b"+" ".join([bin(i)[2:].zfill(8) for i in arr])
    print("a: {}".format(", ".join(["0x{:02X}".format(i) for i in a])))
    print("b: {}".format(", ".join(["0x{:02X}".format(i) for i in b])))
