#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np

np.set_printoptions(formatter={'int': lambda x: "0x{:02X}".format(x)},
                    linewidth=84)

if __name__ == "__main__":
    rnd_data_8bit = np.random.randint(0, 256, (24, )).astype(np.uint8)
    print("rnd_data_8bit:\n{}".format(rnd_data_8bit))

    with open("some_random_data.hex", "wb") as fout:
        rnd_data_8bit.tofile(fout)

    with open("some_random_data.hex", "rb") as fin:
        read_data = np.fromfile(fin, dtype=np.uint8, count=24)

    print("read_data:\n{}".format(read_data))
