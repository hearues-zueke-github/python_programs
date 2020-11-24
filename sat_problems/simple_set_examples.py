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

AMOUNT_SOLUTIONS = 1000

NUM = 10                            # Total number of people
NUM_MALE = 4                        # Total number of males
NUM_DEG = 5                         # Total number of people with university degrees
NUM_MARRIED = 5                     # Total number of married people
NUM_MALE_DEG_SINGLE = 2             # Total number of single males with degree
NUM_FEM_DEG_SINGLE = 1              # Total number of single females with degree
NUM_FEM_AGE_G50 = 3                 # Total number of females with age > 50
NUM_SINGLE_MALE_DEG_AGE_L50 = 1     # Total number of single males with degree with age <= 50

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


def create_cnf_bool_sum_check(var_gen, n, num, preffix_name=''):
    needed_bits = len(bin(n)[2:])
    print("needed_bits: {}".format(needed_bits))
    
    l_var = [var_gen.create_new_var(var_name=f'is_{preffix_name}') for _ in range(0, n)]
    l_var_counter = [[var_gen.create_new_var(var_name=f"{preffix_name}_counter") for _ in range(0, needed_bits)] for _ in range(0, n+1)]
    l_var_counter_remainder = [[var_gen.create_new_var(var_name=f"{preffix_name}_counter_remainder") for _ in range(0, needed_bits-1)] for _ in range(0, n)]

    cnf = []

    # set the first male_counter to zero
    cnf.extend([(-i, ) for i in l_var_counter[0][::-1]])

    l_bits_male = list(map(int, bin(num)[2:].zfill(needed_bits)))
    print("l_bits_male: {}".format(l_bits_male))

    # set the last needed to be number for the male amount counter!
    cnf.extend([(-i if b == 0 else i, ) for i, b in zip(l_var_counter[-1][::-1], l_bits_male)])

    for var, row_x_i, row_x_ip1, row_r in zip(
        l_var,
        l_var_counter[:-1],
        l_var_counter[1:],
        l_var_counter_remainder,
    ):
        # x_i1_0 <-> x_i_0 xor male
        cnf.extend(d_ttrans['xor'](row_x_ip1[0], row_x_i[0], var))
        # r_0 <-> x_i_0 xor male
        cnf.extend(d_ttrans['and'](row_r[0], row_x_i[0], var))

        for var_x_i, var_x_ip1, var_r_i, var_r_ip1 in zip(
            row_x_i[1:-1], row_x_ip1[1:-1], row_r[:-1], row_r[1:]
        ):
            cnf.extend(d_ttrans['xor'](var_x_ip1, var_x_i, var_r_i))
            cnf.extend(d_ttrans['and'](var_r_ip1, var_x_i, var_r_i))

        # last xor
        cnf.extend(d_ttrans['xor'](row_x_ip1[-1], row_x_i[-1], row_r[-1]))

    d = {
        'var': l_var,
        'var_counter': l_var_counter,
        'var_counter_remainder': l_var_counter_remainder,
        'cnf': cnf,
    }

    return d


def create_concat_cnf_bool_sum_check(var_gen, n, num, l_l_var, l_values_mult, preffix_name=''):
    needed_bits = len(bin(n)[2:])
    print("needed_bits: {}".format(needed_bits))
    
    l_var = [var_gen.create_new_var(var_name=f'is_{preffix_name}') for _ in range(0, n)]
    l_var_counter = [[var_gen.create_new_var(var_name=f"{preffix_name}_counter") for _ in range(0, needed_bits)] for _ in range(0, n+1)]
    l_var_counter_remainder = [[var_gen.create_new_var(var_name=f"{preffix_name}_counter_remainder") for _ in range(0, needed_bits-1)] for _ in range(0, n)]

    cnf = []

    # set the first male_counter to zero
    cnf.extend([(-i, ) for i in l_var_counter[0][::-1]])

    l_bits_male = list(map(int, bin(num)[2:].zfill(needed_bits)))
    print("l_bits_male: {}".format(l_bits_male))

    # set the last needed to be number for the male amount counter!
    cnf.extend([(-i if b == 0 else i, ) for i, b in zip(l_var_counter[-1][::-1], l_bits_male)])

    for var, row_x_i, row_x_ip1, row_r in zip(
        l_var,
        l_var_counter[:-1],
        l_var_counter[1:],
        l_var_counter_remainder,
    ):
        # x_i1_0 <-> x_i_0 xor male
        cnf.extend(d_ttrans['xor'](row_x_ip1[0], row_x_i[0], var))
        # r_0 <-> x_i_0 xor male
        cnf.extend(d_ttrans['and'](row_r[0], row_x_i[0], var))

        for var_x_i, var_x_ip1, var_r_i, var_r_ip1 in zip(
            row_x_i[1:-1], row_x_ip1[1:-1], row_r[:-1], row_r[1:]
        ):
            cnf.extend(d_ttrans['xor'](var_x_ip1, var_x_i, var_r_i))
            cnf.extend(d_ttrans['and'](var_r_ip1, var_x_i, var_r_i))

        # last xor
        cnf.extend(d_ttrans['xor'](row_x_ip1[-1], row_x_i[-1], row_r[-1]))

    len_l_l_var = len(l_l_var)
    l_var_combine_columns = [[var_gen.create_new_var(var_name=f"{preffix_name}_combine_columns") for _ in range(0, len_l_l_var-2)] for _ in range(0, n)]
    for var, l_var_c, l_var_combine in zip(l_var, zip(*l_l_var), l_var_combine_columns):
        # len(l_var_c) >= 2!
        if l_var_combine:
            cnf.extend(d_ttrans['and'](l_var_combine[0], l_var_c[0]*l_values_mult[0], l_var_c[1]*l_values_mult[1]))

        for var_c_i, var_c_ip1, var_col, var_mult in zip(l_var_combine[:-1], l_var_combine[1:], l_var_c[2:], l_values_mult[2:]):
            cnf.extend(d_ttrans['and'](var_c_ip1, var_c_i, var_col))

        cnf.extend(d_ttrans['and'](var, l_var_combine[-1], l_var_c[-1]*l_values_mult[-1]))

    d = {
        'var': l_var,
        'var_counter': l_var_counter,
        'var_counter_remainder': l_var_counter_remainder,
        'var_combine_columns': l_var_combine_columns,
        'cnf': cnf,
    }

    return d


if __name__ == '__main__':
    var_gen = VariableGenerator()

    cnf = []

    d_male = create_cnf_bool_sum_check(var_gen=var_gen, n=NUM, num=NUM_MALE, preffix_name='male')
    cnf_add_male = d_male['cnf']
    cnf.extend(cnf_add_male)

    d_deg = create_cnf_bool_sum_check(var_gen=var_gen, n=NUM, num=NUM_DEG, preffix_name='deg')
    cnf_add_deg = d_deg['cnf']
    cnf.extend(cnf_add_deg)

    d_married = create_cnf_bool_sum_check(var_gen=var_gen, n=NUM, num=NUM_MARRIED, preffix_name='married')
    cnf_add_married = d_married['cnf']
    cnf.extend(cnf_add_married)

    d_male_deg_single = create_concat_cnf_bool_sum_check(var_gen=var_gen, n=NUM, num=NUM_MALE_DEG_SINGLE, l_l_var=[d_male['var'], d_deg['var'], d_married['var']], l_values_mult=[1, 1, -1], preffix_name='male_deg_single')
    cnf_add_male_deg_single = d_male_deg_single['cnf']
    cnf.extend(cnf_add_male_deg_single)

    d_fem_deg_single = create_concat_cnf_bool_sum_check(var_gen=var_gen, n=NUM, num=NUM_FEM_DEG_SINGLE, l_l_var=[d_male['var'], d_deg['var'], d_married['var']], l_values_mult=[-1, 1, -1], preffix_name='fem_deg_single')
    cnf_add_fem_deg_single = d_fem_deg_single['cnf']
    cnf.extend(cnf_add_fem_deg_single)

    print("len(cnf): {}".format(len(cnf)))

    l_var = var_gen.l_var
    s_vars = set(l_var)
    
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

    l_var_male = d_male['var']
    l_var_deg = d_deg['var']
    l_var_married = d_married['var']

    l_res_val = l_res_vals[0]

    l_id = list(range(0, NUM))
    l_male = [(l_res_val[i-1]>0)+0 for i in l_var_male]
    l_deg = [(l_res_val[i-1]>0)+0 for i in l_var_deg]
    l_married = [(l_res_val[i-1]>0)+0 for i in l_var_married]

    print("l_male: {}".format(l_male))
    print("l_deg: {}".format(l_deg))
    print("l_married: {}".format(l_married))

    df = pd.DataFrame(data={
            'ID': l_id,
            'Male': l_male,
            'Deg': l_deg,
            'Married': l_married,
        }, columns=['ID', 'Male', 'Deg', 'Married'])

    print("df:\n{}".format(df))

    assert np.sum(df['Male'].values==1) == NUM_MALE
    assert np.sum(df['Deg'].values==1) == NUM_DEG
    assert np.sum(df['Married'].values==1) == NUM_MARRIED

    assert np.sum((df['Male'].values==1)&(df['Deg'].values==1)&(df['Married'].values==0)) == NUM_MALE_DEG_SINGLE
    assert np.sum((df['Male'].values==0)&(df['Deg'].values==1)&(df['Married'].values==0)) == NUM_FEM_DEG_SINGLE
