#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import sys
sys.path.append("../encryption")

import numpy as np

from Utils import pretty_block_printer, clrs

if __name__ == "__main__":

    rnd_data_8bit = np.random.randint(0, 256, (24, )).astype(np.uint8)
    print("{}rnd_data_8bit:{}".format(clrs.lyb, clrs.rst))
    pretty_block_printer(rnd_data_8bit, 8, rnd_data_8bit.shape[0])

    with open("some_random_data.hex", "wb") as fout:
        rnd_data_8bit.tofile(fout)

    with open("some_random_data.hex", "rb") as fin:
        read_data = np.fromfile(fin, dtype=np.uint8, count=24)
        # read_data = np.fromfile(fin, dtype=np.uint16, count=12)

    # # To change from big to little endian, and vice versa
    # read_data = (read_data>>8)|(read_data<<8)

    print("{}read_data:{}".format(clrs.lyb, clrs.rst))
    pretty_block_printer(read_data, 8, read_data.shape[0])

    # with open("/dev/sdf", "rb") as fin:
    #     data = np.fromfile(fin, dtype=np.uint8, count=512)

    # print("{}data:{}".format(clrs.lyb, clrs.rst))
    # pretty_block_printer(data, 8, data.shape[0])
