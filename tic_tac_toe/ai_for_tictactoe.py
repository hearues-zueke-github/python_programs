#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9.4 -i

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

import numpy as np
import pandas as pd
import multiprocessing as mp

from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from memory_tempfile import MemoryTempfile
from shutil import copyfile

import pymongo
from bson.objectid import ObjectId

from pprint import pprint
from typing import List, Set, Tuple, Dict, Union

from PIL import Image

import importlib.util as imp_util
spec = imp_util.spec_from_file_location("utils", "../utils.py")
utils = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils)

mkdirs = utils.mkdirs

spec = imp_util.spec_from_file_location("utils_multiprocessing_manager", "../utils_multiprocessing_manager.py")
utils_multiprocessing_manager = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils_multiprocessing_manager)

MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"
HOME_DIR = os.path.expanduser("~")+"/"
TEMP_DIR = MemoryTempfile().gettempdir()+"/"

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs/'
mkdirs(OBJS_DIR_PATH)

def get_tictactoe_collection():
    myclient = pymongo.MongoClient("mongodb://myuser:myuser#123@localhost:27017/db_own")
    mydb = myclient["db_own"]
    mycol = mydb['tictactoe']

    return myclient, mycol


def get_fill_random_field(n: int, amount_symbols: int, l_pos: List[Tuple[int, int]], l_symbols_nr: List[int]) -> np.ndarray:
    field = np.zeros((n, n), dtype=np.uint8)

    arr_rnd = np.random.permutation(np.arange(0, len(l_pos)))

    arr_differences = np.zeros((amount_symbols, ), dtype=np.int64)
    arr_differences[:] = n**2 // amount_symbols
    arr_differences[:n**2 % amount_symbols] += 1

    l_indices = [0] + np.cumsum(arr_differences).tolist()

    for symbol_nr, i1, i2 in zip(l_symbols_nr, l_indices[:-1], l_indices[1:]):
        for i in arr_rnd[i1:i2]:
            t_pos = l_pos[i]
            x, y = t_pos
            field[y, x] = symbol_nr

    return field


def add_new_arr_stats(n: int, amount_symbols: int, max_iters: int) -> Union[np.ndarray, None]:
    myclient, mycol = get_tictactoe_collection()
    l_find = list(mycol.find({'n': n, 'amount_symbols': amount_symbols}))
    myclient.close()

    if len(l_find) > 0:
        print('Already exists for: n: {}, amount_symbols: {}'.format(n, amount_symbols))
        sys.exit()

    # max_iters = 10000

    # def get_arr_stats(max_iters: int) -> np.ndarray:
    l_symbols_nr: List[int] = list(range(1, amount_symbols+1))
    d_next_symbol_nr: Dict[int, int] = {sym1: sym2 for sym1, sym2 in zip(l_symbols_nr, l_symbols_nr[1:]+l_symbols_nr[:1])}

    print("l_symbols_nr: {}".format(l_symbols_nr))
    print("d_next_symbol_nr: {}".format(d_next_symbol_nr))

    s_pos: Set[Tuple[int, int]] = set([(x, y) for y in range(0, n) for x in range(0, n)])
    l_pos: List[Tuple[int, int]] = sorted(s_pos)

    def combine_list_positions(l_pos_list: List[Tuple[int, ...]]) -> Tuple[List[int], List[int]]:
        l_y = []
        l_x = []

        for l_y_next, l_x_next in sorted(set(l_pos_list)):
            l_y += list(l_y_next)
            l_x += list(l_x_next)

        return (l_y, l_x)

    # The lengths of the different cluster sizes! for n=3 and in 2-d there are
    # only lengths 1, 2 and 3.
    d_pos_list_clusters: Dict[int, Tuple[List[int], List[int]]] = {}

    for i in range(1, n+1):
        d_pos_list_clusters[i] = combine_list_positions(
            [(tuple(y+j for j in range(0, i)), tuple(x for _ in range(0, i))) for y in range(0, n-i+1) for x in range(0, n)] +
            [(tuple(y for _ in range(0, i)), tuple(x+j for j in range(0, i))) for y in range(0, n) for x in range(0, n-i+1)] +
            [(tuple(y+j for j in range(0, i)), tuple(x+j for j in range(0, i))) for y in range(0, n-i+1) for x in range(0, n-i+1)] +
            [(tuple(y+j for j in range(0, i)), tuple(x+j for j in range(i-1, -1, -1))) for y in range(0, n-i+1) for x in range(0, n-i+1)]
        )

    arr_stats = np.zeros((amount_symbols, amount_symbols), dtype=np.int64)
    
    def get_field_statistics(field: np.ndarray) -> pd.DataFrame:
        d_data = {f'{i}': [] for i in l_symbols_nr} | {'length': []}
        for i, l_y_x in d_pos_list_clusters.items():
            arr = field[l_y_x].reshape((-1, i))

            d_data['length'].append(i)
            for j in l_symbols_nr:
                d_data[f'{j}'].append(np.sum(np.all(arr==j, axis=1)))

        df = pd.DataFrame(data=d_data)
        df = df[['length']+[f'{i}' for i in l_symbols_nr]]

        return df

    def add_too_arr_stats(df: pd.DataFrame, arr_stats: np.ndarray) -> None:
        l_values_symbol_nr_not_sorted = [(tuple(l[::-1]), symbols_nr) for l, symbols_nr in zip(df.values.T.tolist()[1:], l_symbols_nr)]
        l_values_symbol_nr = sorted(l_values_symbol_nr_not_sorted, reverse=True)
        d_ranks = {symbol_nr: rank for rank, (_, symbol_nr) in enumerate(l_values_symbol_nr, 0)}

        for symbol_nr, rank in d_ranks.items():
            arr_stats[symbol_nr-1, rank] += 1

    for iter_i in range(1, max_iters+1):
        field = get_fill_random_field(n, amount_symbols, l_pos, l_symbols_nr)
        df = get_field_statistics(field)

        add_too_arr_stats(df, arr_stats)
        if iter_i % 1000 == 0:
            print("iter_i: {}".format(iter_i))
            print("arr_stats:\n{}".format(arr_stats))

        # return arr_stats

    # arr_stats = get_arr_stats(max_iters)

    print("n: {}".format(n))
    print("amount_symbols: {}".format(amount_symbols))
    print("max_iters: {}".format(max_iters))
    print("arr_stats:\n{}".format(arr_stats))

    myclient, mycol = get_tictactoe_collection()
    mycol.insert_one({'n': n, 'amount_symbols': amount_symbols, 'l_stats': arr_stats.tolist()})
    myclient.close()

    return arr_stats


def convert_mongo_list_to_df(l_find):
    # get all possible fields
    s_column = set()
    for value in l_find:
        s_column |= set(value.keys())

    d_data = {column: [] for column in s_column}

    for value in l_find:
        s_column_include = set(value.keys())
        s_column_not_include = s_column - s_column_include

        for column in s_column_include:
            d_data[column].append(value[column])

        for column in s_column_not_include:
            d_data[column].append(None)

    df = pd.DataFrame(data=d_data, dtype=object)

    return df


def prepare_df_tictactoe_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[['amount_symbols', 'n', 'l_stats']]


def get_tictactoe_statistics() -> pd.DataFrame:
    myclient, mycol = get_tictactoe_collection()
    l_find = list(mycol.find({}, {'_id': 0}))
    myclient.close()

    df = convert_mongo_list_to_df(l_find)

    df = df[['amount_symbols', 'n', 'l_stats']]
    df.sort_values(by=['amount_symbols', 'n'], inplace=True)

    df['arr_stats'] = pd.Series([np.array(l_stats, dtype=np.int64) for l_stats in df['l_stats'].values], index=df.index)
    df['arr_first_place'] = pd.Series([np.round(arr_stats[:, 0] / np.sum(arr_stats[:, 0]) * 100, 2) for arr_stats in df['arr_stats'].values], index=df.index)

    return df


if __name__ == '__main__':
    # myclient, mycol = get_tictactoe_collection()
    # l_find = list(mycol.find({}, {'_id': 0}))
    # # myclient.close()
    
    # # mycol.delete_many({}) # drop whole collection

    df = get_tictactoe_statistics()

    sys.exit()

    # argv = sys.argv

    # n: int = int(argv[1])
    # amount_symbols: int = int(argv[2])
    max_iters: int = 10000

    # arr_stats = add_new_arr_stats(n, amount_symbols, max_iters)

    def worker_thread(n, amount_symbols, max_iters):
        np.random.seed()
        add_new_arr_stats(n, amount_symbols, max_iters)

    mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())

    mult_proc_mng.define_new_func('func_worker_thread', worker_thread)

    min_amount_symbols = 11
    max_amount_symbols = 20
    l_arguments = [(n, amount_symbols, max_iters) for amount_symbols in range(min_amount_symbols, max_amount_symbols+1) for n in range(amount_symbols, max_amount_symbols+1)]
    l_ret = mult_proc_mng.do_new_jobs(
        ['func_worker_thread']*len(l_arguments),
        l_arguments,
    )
    print("len(l_ret): {}".format(len(l_ret)))

    del mult_proc_mng

    df = get_tictactoe_statistics()
    print("df:\n{}".format(df))
