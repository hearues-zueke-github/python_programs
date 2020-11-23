#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import traceback

import multiprocessing as mp
import numpy as np
import pandas as pd

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from pprint import pprint
from shutil import copyfile

sys.path.append('..')
from utils import mkdirs
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

from pysat.solvers import Glucose3

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

AMOUNT_SOLUTIONS = 10000


class VariableGenerator(Exception):
    def __init__(self):
        self.i = 1
        self.var_name_idx_agg = 0

        self.l_var = []
        self.l_var_name_idx = []
        self.d_var_name_idx = {}


    def create_new_var(self, var_name):
        i = self.i
        self.i += 1
        self.l_var.append(i)
        var_name_idx = self.get_var_name_idx(var_name=var_name)
        self.l_var_name_idx.append(var_name_idx)
        return i


    def get_var_name_idx(self, var_name):
        if var_name in self.d_var_name_idx:
            return self.d_var_name_idx[var_name]

        var_name_idx_agg = self.var_name_idx_agg
        self.var_name_idx_agg += 1
        self.d_var_name_idx[var_name] = var_name_idx_agg
        return var_name_idx_agg


# y, x1, x2 are the variable number!
def tseytin_transform_and(y, x1, x2):
    return [(-x1, -x2, y), (x1, -y), (x2, -y)]


def tseytin_transform_nand(y, x1, x2):
    return [(-x1, -x2, -y), (x1, y), (x2, y)]


def tseytin_transform_or(y, x1, x2):
    return [(x1, x2, -y), (-x1, y), (-x2, y)]


def tseytin_transform_nor(y, x1, x2):
    return [(x1, x2, y), (-x1, -y), (-x2, -y)]


def tseytin_transform_not(y, x1):
    return [(-x1, -y), (x1, y)]


def tseytin_transform_xor(y, x1, x2):
    return [(-x1, -x2, -y), (x1, x2, -y), (x1, -x2, y), (-x1, x2, y)]


def tseytin_transform_xnor(y, x1, x2):
    return [(-x1, -x2, y), (x1, x2, y), (x1, -x2, -y), (-x1, x2, -y)]


d_ttrans ={
    'and': tseytin_transform_and,
    'nand': tseytin_transform_nand,
    'or': tseytin_transform_or,
    'nor': tseytin_transform_nor,
    'not': tseytin_transform_not,
    'xor': tseytin_transform_xor,
    'xnor': tseytin_transform_xnor,
}

if __name__ == '__main__':
    var_gen = VariableGenerator()

    AMOUNT_N = 10
    AMOUNT_MALE = 5

    l_var_is_male = [var_gen.create_new_var(var_name="is_male") for _ in range(0, AMOUNT_N)]

    needed_bits = len(bin(AMOUNT_N)[2:])
    print("needed_bits: {}".format(needed_bits))

    print("l_var_is_male: {}".format(l_var_is_male))

    l_var_male_counter = [[var_gen.create_new_var(var_name="male_counter") for _ in range(0, needed_bits)] for _ in range(0, AMOUNT_N+1)]
    l_var_male_counter_remainder = [[var_gen.create_new_var(var_name="male_counter_remainder") for _ in range(0, needed_bits-1)] for _ in range(0, AMOUNT_N)]
    print("l_var_male_counter: {}".format(l_var_male_counter))
    print("l_var_male_counter_remainder: {}".format(l_var_male_counter_remainder))

    cnf = []

    # set the first male_counter to zero
    cnf.extend([(-i, ) for i in l_var_male_counter[0][::-1]])

    l_bits_male = list(map(int, bin(AMOUNT_MALE)[2:].zfill(needed_bits)))
    print("l_bits_male: {}".format(l_bits_male))

    # set the last needed to be number for the male amount counter!
    cnf.extend([(-i if b == 0 else i, ) for i, b in zip(l_var_male_counter[-1][::-1], l_bits_male)])

    for var_male, row_x_i, row_x_ip1, row_r in zip(
        l_var_is_male,
        l_var_male_counter[:-1],
        l_var_male_counter[1:],
        l_var_male_counter_remainder,
    ):
        # x_i1_0 <-> x_i_0 xor male
        cnf.extend(d_ttrans['xor'](row_x_ip1[0], row_x_i[0], var_male))
        # r_0 <-> x_i_0 xor male
        cnf.extend(d_ttrans['and'](row_r[0], row_x_i[0], var_male))

        for var_x_i, var_x_ip1, var_r_i, var_r_ip1 in zip(
            row_x_i[1:-1], row_x_ip1[1:-1], row_r[:-1], row_r[1:]
        ):
            cnf.extend(d_ttrans['xor'](var_x_ip1, var_x_i, var_r_i))
            cnf.extend(d_ttrans['and'](var_r_ip1, var_x_i, var_r_i))

        # last xor
        cnf.extend(d_ttrans['xor'](row_x_ip1[-1], row_x_i[-1], row_r[-1]))

    # e.g. add some known is_male variables as an example!
    # cnf.extend([(l_var_is_male[i], ) for i in [0, 1, 2, 3]])

    print("Print the whole cnf form:")
    pprint(cnf)

    # sys.exit(0)

    l_var = var_gen.l_var
    s_vars = set(l_var)
    
    # cnf = [
    #     (-1, -2, 3),
    #     (1, -3),
    #     (2, -3),
    #     # (1, ),
    # ]

    def get_all_variables(l_vars_true):
        s_vars_false = s_vars - set(l_vars_true)
        return [(lambda x: -x if x in s_vars_false else x)(i) for i in l_var]

    with Glucose3(bootstrap_with=cnf) as m:
        print(m.solve())
        print(list(m.get_model()))
        models = [m for m, _ in zip(m.enum_models(), range(0, AMOUNT_SOLUTIONS))]
        # idxs = [i-1 for i in models[np.random.randint(0, len(models))] if i > 0]
        l_res_idxs = [[i for i in m if i > 0] for m in models]
        l_res_vals = [get_all_variables(l_idxs) for l_idxs in l_res_idxs]

    print("len(l_res_vals): {}".format(len(l_res_vals)))
    print("l_res_vals[0]: {}".format(l_res_vals[0]))
