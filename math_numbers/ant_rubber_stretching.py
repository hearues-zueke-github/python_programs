#! /usr/bin/python3

import numpy as np

if __name__ == "__main__":
    l0 = 1.
    d0 = 0.06

    print("l0: {}".format(l0))
    print("d0: {}".format(d0))

    l = l0
    d = d0

    for i in range(1, 10000000):
        l1 = l+l0
        d = d*l1/l+d0

        l = l1

        if d >= l:
            print("STOP!")
            print("i: {}".format(i))
            break
    print("l: {}".format(l))
    print("d: {}".format(d))
