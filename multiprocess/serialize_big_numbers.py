#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import marshal
import pickle
import os

import multiprocessing as mp
import numpy as np

from dotmap import DotMap
from multiprocessing import Process, Queue

from time import time

if __name__ == "__main__":
    from os.path import expanduser
    home = expanduser("~")
    print("home: {}".format(home))

    path_files = home+"/Documents/"

    dm = DotMap()
    dm.num_a = 3**1000
    dm.num_b = 3**1001
    dm.num_c = 3**1002

    with open(path_files+"big_numbers.pkl", "wb") as fout:
        dill.dump(dm, fout)
