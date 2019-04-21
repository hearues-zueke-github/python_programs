#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

if __name__ == "__main__":
    n = 10
    arr = np.random.randint(0, 20, (n, ))
    print("arr: {}".format(arr))

    check = np.hstack(((1, ), (arr[1:]>=arr[:-1])+0))
    print("check: {}".format(check))

    chuncks = np.hstack(((0, ), np.where(check[1:]!=check[:-1])[0]+1, (check.shape[0], )))
    print("chuncks: {}".format(chuncks))
