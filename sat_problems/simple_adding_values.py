#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

# -*- coding: utf-8 -*-

import os
import random
import sys

from pysat.solvers import Glucose3

from copy import deepcopy

from time import time
from functools import reduce

from PIL import Image

import numpy as np

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

from pprint import pprint
from typing import List

from cnf import CNF

if __name__ == "__main__":
    cnf_obj = CNF()
    
    l_v = cnf_obj.get_new_variables(amount=5)
    print(f"l_v: {l_v}")

    cnfs_part_count_sum = cnf_obj.add_count_sum(l_v=l_v, bits=4, num=3)
    cnf_obj.extend_cnfs(cnfs_part_count_sum)

    models_amount = 16
    with Glucose3(bootstrap_with=cnf_obj.cnfs) as m:
        print(m.solve())
        print(list(m.get_model()))
        models = [(i, m) for m, i in zip(m.enum_models(), range(0, models_amount))]

    for m in models:
        l = m[1]

        l1 = l[:5]
        l2 = l[-4:]

        s1 = sum(v>0 for v in l1)
        s2 = np.sum((np.array(l2) > 0) * 2**np.arange(0, 4))

        print(f"l1: {l1}, l2: {l2}, s1: {s1}, s2: {s2}")
        assert s1 == s2
