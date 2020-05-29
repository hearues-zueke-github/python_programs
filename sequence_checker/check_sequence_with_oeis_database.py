#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import pdb
import sys

# Needed for DataFrame (databases etc.)
import numpy as np
import pandas as pd
import pandas.io.sql as sql

from functools import reduce

from dotmap import DotMap

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

USER_HOME_PATH = os.path.expanduser('~')+'/'
print("USER_HOME_PATH: {}".format(USER_HOME_PATH))

if __name__ == "__main__":
    file_name = "stripped"
    file_path = USER_HOME_PATH+"Documents/{file_name}".format(file_name=file_name)
    if not os.path.exists(file_path):
        print("Please add the 'stripped' file into your '~/Documents/' folder!")

    with open(file_path, "r") as f:
        lines = f.readlines()
    def prepare_line(line):
        if not ",\n" in line:
            return None
        line = line.replace("\n", "")
        if line[-1] == ",":
            line = line[:-1]
        A_code, number_sequence = line.split(" ,")
        return [A_code, np.array(list(map(int, number_sequence.split(","))), dtype=object)]
    lines = [prepare_line(line) for line in lines]
    lines = list(filter(lambda x: x != None, lines))
    print("len(lines): {}".format(len(lines)))

    A_codes, number_sequences = list(zip(*lines))
    A_codes = np.array(A_codes)
    number_sequences = np.array(number_sequences)

    idxs = [i for i, ns in enumerate(number_sequences, 0) if np.any((ns<2**63) & (ns >= -2**63))]

    print("len(idxs): {}".format(len(idxs)))

    A_codes = A_codes[idxs]
    number_sequences = number_sequences[idxs]

    lens = np.array(list(map(len, number_sequences)))
    u, c = np.unique(lens, return_counts=True)

    idxs_sort = np.flip(np.argsort(u))
    u = u[idxs_sort]
    c = c[idxs_sort]


