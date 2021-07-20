#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9.5 -i

# -*- coding: utf-8 -*-

import dill
import gzip
import hashlib
import os
import string
import sys
import inspect
import textwrap

from typing import List, Dict, Set, Mapping, Any, Tuple

# import tempfile
from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

from datetime import datetime
from collections import defaultdict
from copy import deepcopy
from dotmap import DotMap
from operator import itemgetter

from pprint import pprint

from os.path import expanduser

import itertools

import multiprocessing as mp

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

from PIL import Image

import numpy as np
import pandas as pd

sys.path.append('..')
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PATH_ROOT_DIR, "../utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_function', path=os.path.join(PATH_ROOT_DIR, "../utils_function.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_cluster', path=os.path.join(PATH_ROOT_DIR, "../clustering/utils_cluster.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PATH_ROOT_DIR, "../utils_multiprocessing_manager.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='cell_matrix_unique', path=os.path.join(PATH_ROOT_DIR, "../cell_math/cell_matrix_unique.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_serialization', path=os.path.join(PATH_ROOT_DIR, "../utils_serialization.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='bit_automaton', path=os.path.join(PATH_ROOT_DIR, "bit_automaton.py")))

mkdirs = utils.mkdirs

copy_function = utils_function.copy_function

MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

get_d_l_t_2d_cells = cell_matrix_unique.get_d_l_t_2d_cells

get_pkl_gz_obj = utils_serialization.get_pkl_gz_obj
save_pkl_gz_obj = utils_serialization.save_pkl_gz_obj
load_pkl_gz_obj = utils_serialization.load_pkl_gz_obj

BitAutomaton = bit_automaton.BitAutomaton

DIR_PATH_SAVE_IMAGES = os.path.join(TEMP_DIR, 'save_images/')

def range_gen_old(g, n):
    try:
        i = 0
        if i >= n:
            return
        v = next(g)
        while v != None:
            yield v
            v = next(g)
            i += 1
            if i >= n:
                break
    except StopIteration:
        return

def range_gen(g, n):
    try:
        i = 0
        while ((i := i + 1) <= n) and ((v := next(g)) != None):
            yield v
    except StopIteration:
        return

    # return

def gen_new_gen_nr1():
    return iter(range(0, 100))

l1 = list(range_gen_old(gen_new_gen_nr1(), 0))
l2 = list(range_gen(gen_new_gen_nr1(), 0))
assert l1 == l2

l1 = list(range_gen_old(gen_new_gen_nr1(), 1))
l2 = list(range_gen(gen_new_gen_nr1(), 1))
assert l1 == l2

l1 = list(range_gen_old(gen_new_gen_nr1(), 40))
l2 = list(range_gen(gen_new_gen_nr1(), 40))
assert l1 == l2

l1 = list(range_gen_old(gen_new_gen_nr1(), 200))
l2 = list(range_gen(gen_new_gen_nr1(), 200))
assert l1 == l2

# sys.exit()


def prepare_functions(funcs_str: str, frame: int) -> Tuple[List[Any], Any, Any, int]:
    bit_automaton = BitAutomaton().init_vals(h=10, w=8, frame=frame, frame_wrap=True, l_func=[], func_inv=None, func_rng=None)
    
    d_vars = bit_automaton.d_vars
    field_size = bit_automaton.field_size
    d_func = {}
    exec(funcs_str, d_vars, d_func)
    print("d_func.keys(): {}".format(d_func.keys()))

    d_func_keys = d_func.keys()
    assert 'rng' in d_func_keys
    assert 'inv' in d_func_keys
    assert 'l_func' in d_func_keys
    assert 'start_seed' in d_func_keys

    func_inv = copy_function(d_func['inv'])

    d_vars['inv'] = func_inv

    # test if each function starting with 'fun_' is returning a boolean array!
    l_func_name = []
    for func_name in d_func.keys():
        if func_name[:4] != 'fun_':
            continue

        l_func_name.append(func_name)

        # print("func_name: {}".format(func_name))
        v = d_func[func_name]()
        assert v.dtype == bool
        assert v.shape == field_size

    # check, if every function name is appearing starting from 0 to upwards in ascending order!
    assert np.all(np.diff(np.sort([int(v.replace('fun_', '')) for v in l_func_name])) == 1)

    l_func = [d_func[func_name] for func_name in l_func_name]
    # s_func_nr = set(range(0, len(l_func_name)))
    func_inv = d_func['inv']
    func_rng = d_func['rng']

    start_seed = d_func['start_seed']

    return l_func, func_inv, func_rng, start_seed


def create_random_function_data(frame: int = 1, frame_wrap: bool = True) -> DotMap:
    bit_automaton = BitAutomaton().init_vals(h=10, w=8, frame=frame, frame_wrap=frame_wrap, l_func=[], func_inv=None,
                                             func_rng=None)

    funcs_str_inv = "def inv(x):\n return ~x\n"
    funcs_str_rng = """
    def rng(seed=0):
     a = seed
     while True:
      a = ((a + 19) ^ 0x2343) % 15232
      yield a % 13

    """

    l_digits = string.digits
    l_var = [v for v in bit_automaton.d_vars.keys() if not any([c in l_digits for c in v])]
    del bit_automaton

    print("l_var: {}".format(l_var))

    def get_random_function_str_body_dm(
            func_nr: int,
            l_var: List[str],
            n_and_min: int,
            n_and_max: int,
            n_or_min: int,
            n_or_max: int,
    ) -> str:
        l_vars_inv = [f'inv({v})' for v in l_var]

        arr_var = np.vstack((l_var, l_vars_inv))
        arr_var_i = np.arange(0, arr_var.shape[1])
        arr_var_row = np.arange(0, 2)

        func_name = f'fun_{func_nr}'
        # s = f'def {func_name}():\n'
        s = ''
        l_new_var = []
        l_l_chosen_var = []
        l_arr_var_names = []
        l_formel_and = []
        n_or = np.random.randint(n_or_min, n_or_max + 1)
        for i_or in range(0, n_or):
            new_var = f'a_{i_or}'
            l_new_var.append(new_var)

            n_and = np.random.randint(n_and_min, n_and_max + 1)
            arr_chosen_var = np.unique(np.random.choice(arr_var_i, (n_and,)))
            arr_inv_var = np.random.choice(arr_var_row, (arr_chosen_var.shape[0],))

            arr_var_names = np.sort(arr_var[arr_inv_var, arr_chosen_var])
            l_arr_var_names.append(arr_var_names)

            formel_and = ' & '.join(arr_var_names)
            l_formel_and.append(formel_and)

        l_former_and_sorted = sorted(set(l_formel_and))
        for new_var, formel_and in zip(l_new_var, l_formel_and):
            s += f' {new_var} = {formel_and}\n'
            l_l_chosen_var.append((arr_chosen_var, arr_inv_var))

        last_formel = ' | '.join(l_new_var)
        s += f' return {last_formel}\n'
        return DotMap(locals(), dynamic_=None)

    # TODO: save the data to a file too!
    n_func_min = 1
    n_func_max = 1
    n_or_min = 2
    n_or_max = 10
    n_and_min = 2
    n_and_max = 6

    n_func = np.random.randint(n_func_min, n_func_max + 1)
    # TODO: create many random images (e.g. 5 times) with the same function!
    l_dm_local = [
        get_random_function_str_body_dm(
            func_nr=i,
            l_var=l_var,
            n_and_min=n_and_min,
            n_and_max=n_and_max,
            n_or_min=n_or_min,
            n_or_max=n_or_max,
        ) for i in range(0, n_func)]
    l_func_str_body = [dm.s for dm in l_dm_local]

    l_func_str_body_sorted = sorted(set(l_func_str_body))
    l_func_name = [dm.func_name for dm in l_dm_local[:len(l_func_str_body_sorted)]]
    l_func_str_sorted = [f'def {func_name}():\n{s}' for func_name, s in zip(l_func_name, l_func_str_body_sorted)]

    func_str = ', '.join([func_name for func_name in l_func_name])
    funcs_str_func = '\n'.join(l_func_str_sorted)

    func_list_str = 'l_func = [{}]'.format(func_str)

    funcs_str = '\n'.join([
        inspect.cleandoc(funcs_str_rng) + '\n',
        funcs_str_inv,
        funcs_str_func,
        func_list_str,
        'start_seed = 0',
        '\n',
    ])

    return locals()


def create_new_automaton(d_function_data: Dict[Any, Any], l_t_2d_cells: Any) -> None:
    funcs_str = d_function_data['funcs_str']
    l_func_str_sorted = d_function_data['l_func_str_sorted']
    frame = d_function_data['frame']
    frame_wrap = d_function_data['frame_wrap']

    # sys.exit()

    # TODO: create some random functions too!

    l_func, func_inv, func_rng, start_seed = prepare_functions(funcs_str=funcs_str, frame=frame)


    # path_images = PATH_ROOT_DIR + 'images/'

    # file_name = 'nature_trees_river_84761_800x600.jpg'
    # image_path = path_images + file_name

    # img = Image.open(image_path)
    # pix = np.array(img)

    pix_height = 100
    pix_width = 150
    pix_orig = np.random.randint(0, 255, (pix_height, pix_width, 3), dtype=np.uint8)

    def extract_bits_from_pix(pix : np.ndarray) -> np.ndarray:
        pix_bits = np.array([(channel>>i)&0x1 for channel in pix.transpose(2, 0, 1) for i in range(7, -1, -1)], dtype=np.uint8)
        return pix_bits
        # return pix_bits.transpose(1, 2, 0)

    # def extract_rgb_from_pix(pix : np.ndarray) -> np.ndarray:
    #     return pix.transpose(2, 0, 1)

    pix_bits = extract_bits_from_pix(pix_orig)

    pix = pix_bits[0] * 255
    # pix = pix_bits[:1]

    h, w = pix_bits.shape[1:]
    # h, w = pix_bits.shape[:2]
    # sys.exit()

    # folder_name = 'test_other'
    folder_name_suffix = '{}_{}'.format(
        datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M:%S'),
        ''.join(np.random.choice(list(string.hexdigits), (8, ))).upper(),
    )
    folder_name = f'test_other_{folder_name_suffix}'
    dir_path_images = os.path.join(DIR_PATH_SAVE_IMAGES, folder_name)
    # dir_path_images = os.path.join(TEMP_DIR, 'save_images/{}/'.format(file_name.split('.')[0]))
    mkdirs(dir_path_images)
    
    dir_path_combined_images = os.path.join(DIR_PATH_SAVE_IMAGES, 'combined_images')
    mkdirs(dir_path_combined_images)

    dir_path_combined_images_xor = os.path.join(DIR_PATH_SAVE_IMAGES, 'combined_images_xor')
    mkdirs(dir_path_combined_images_xor)

    file_path_funcs = os.path.join(dir_path_images, 'test_functions_python.py')
    with open(file_path_funcs, 'w') as f:
        f.write(funcs_str)

    arr_bits = pix_bits[:1]
    # arr_bits = np.array(
    #     [[list(itertools.chain(*[list(map(int, bin(b)[2:].zfill(8))) for b in v])) for v in row] for row in pix],
    #     dtype=np.uint8,
    # ).transpose(2, 0, 1)

    amount_bit_automaton = arr_bits.shape[0]

    l_bit_automaton = [BitAutomaton().init_vals(h=h, w=w, frame=frame, frame_wrap=frame_wrap, l_func=l_func, func_inv=func_inv, func_rng=func_rng) for _ in range(0, amount_bit_automaton)]

    for bit_automaton, bits in zip(l_bit_automaton, arr_bits):
        bit_automaton.set_field(bits.astype(bool))

    def convert_bit_field_to_pix_1_bit(l_bit_automaton):
        assert len(l_bit_automaton) == 1 # 1 bits

        arr = (l_bit_automaton[0]<<0).astype(np.uint8) * 255
        return arr

    def convert_bit_field_to_pix_8_bit(l_bit_automaton):
        assert len(l_bit_automaton) == 8 # 8 bits

        arr = np.sum([bit_automaton<<j for j, bit_automaton in zip(range(7, -1, -1), l_bit_automaton)], axis=0).astype(np.uint8)
        return arr

    def convert_bit_field_to_pix_24_bit(l_bit_automaton):
        assert len(l_bit_automaton) == 24 # 24 bits

        arr = np.array([np.sum([bit_automaton<<j for j, bit_automaton in zip(range(7, -1, -1), l_bit_automaton[8*i:8*(i+1)])], axis=0) for i in range(0, 3)], dtype=np.uint8)
        return arr.transpose(1, 2, 0)

    convert_bit_field_to_pix = convert_bit_field_to_pix_1_bit
    # convert_bit_field_to_pix = convert_bit_field_to_pix_8_bit
    # convert_bit_field_to_pix = convert_bit_field_to_pix_24_bit

    pix2 = convert_bit_field_to_pix(l_bit_automaton[:1])
    assert np.all(pix2 == pix)

    print("i: {}".format(0))
    file_path = os.path.join(dir_path_images, '{:04}.png'.format(0))
    Image.fromarray(pix2).save(file_path)
    
    l_pix = [pix2]

    # cols = 5
    # rows = 10

    cols = 10
    rows = 18
    iterations_amount = cols * rows
    l_func_nr = list(range(0, len(l_func)))
    amount_function_mod = len(l_func_nr)
    rng = func_rng(seed=start_seed)
    # for i in range(1, 100):
    for i in range(1, iterations_amount):
        func_nr = l_func_nr[next(rng) % amount_function_mod]
        print("i: {}, func_nr: {}".format(i, func_nr))

        for bit_automaton in l_bit_automaton:
            # bit_automaton.execute_func(5)
            bit_automaton.execute_func(func_nr)
        pix2 = convert_bit_field_to_pix(l_bit_automaton)
        file_path = os.path.join(dir_path_images, '{:04}.png'.format(i))
        # Image.fromarray(pix2).save(file_path)

        l_pix.append(pix2)

    arr_pixs = np.array(l_pix) // 255

    # find all unique v_2d_cells
    s_2d_cells = set()
    for t_2d_cells in l_t_2d_cells:
        for v_2d_cells in t_2d_cells:
            if v_2d_cells not in s_2d_cells:
                s_2d_cells.add(v_2d_cells)

    # create all possible arr_pixs_roll arrays
    d_arr_pixs_roll = {}
    for dy, dx in s_2d_cells:
        d_arr_pixs_roll[(dy, dx)] = np.roll(np.roll(arr_pixs, dy, 1), dx, 2)

    # get from all the cols * rows images the cluster_roll sum
    l_sum_pix_cluster_roll = []
    arr_pix_temp = np.zeros((rows * cols, pix_height, pix_width), dtype=np.uint8)
    for t_2d_cells in l_t_2d_cells:
        arr_pix_temp[:] = 1
        for v_2d_cells in t_2d_cells:
            arr_pix_temp &= d_arr_pixs_roll[v_2d_cells]
        l_sum_pix_cluster_roll.append(np.sum(np.sum(arr_pix_temp, 2), 1))

    arr_sum_pix_cluster_roll = np.array(l_sum_pix_cluster_roll).T

    arr_historic_ranges = np.zeros((arr_pixs.shape[0], 2), dtype=np.int_)
    for i, pix in enumerate(arr_pixs, 0):
        sum_0 = np.sum(np.equal(pix, 0))
        sum_1 = np.sum(np.equal(pix, 1))
        arr_historic_ranges[i] = [sum_0, sum_1]

    # TODO: find _dynamic in other files too! correct this in every files!
    dm_obj = DotMap(_dynamic=None)
    dm_obj['cols'] = cols
    dm_obj['rows'] = rows
    dm_obj['pix_height'] = pix_height
    dm_obj['pix_width'] = pix_width

    dm_obj['l_t_2d_cells'] = l_t_2d_cells
    dm_obj['l_sum_pix_cluster_roll'] = l_sum_pix_cluster_roll
    dm_obj['arr_sum_pix_cluster_roll'] = arr_sum_pix_cluster_roll

    dm_obj['frame'] = frame
    dm_obj['frame_wrap'] = frame_wrap
    # dm_obj['l_bit_automaton'] = l_bit_automaton
    dm_obj['arr_pixs'] = arr_pixs
    dm_obj['arr_historic_ranges'] = arr_historic_ranges
    dm_obj['func_str'] = l_func_str_sorted[0]
    dm_obj['l_func_str_sorted'] = l_func_str_sorted
    dm_obj['func_str_hash'] = hashlib.sha512(dm_obj['func_str'].encode()).hexdigest()
    dm_obj['_version'] = utils_cluster.__version__
    dm_obj['d_function_data'] = d_function_data

    with gzip.open(os.path.join(dir_path_images, utils_cluster.dm_obj_file_name), 'wb') as f:
        dill.dump(dm_obj, f)

    def combine_all_pix(l_pix, w_space_horizontal=10, h_space_vertical=10):
        h_space_horizontal = l_pix[0].shape[0]
        arr_space_horizontal = np.zeros((h_space_horizontal, w_space_horizontal), dtype=np.uint8) + 0x80

        def combine_l_pix_horizontal(l_pix_part : List[np.ndarray]) -> np.ndarray:
            l = [l_pix_part[0]]
            for pix in l_pix_part[1:]:
                l.append(arr_space_horizontal)
                l.append(pix)
            return np.hstack(l)

        l_pix_horizontal = [combine_l_pix_horizontal(l_pix[cols*i:cols*(i+1)]) for i in range(0, rows)]

        w_space_vertical = l_pix_horizontal[0].shape[1]
        # h_space_vertical = 10
        arr_space_vertical = np.zeros((h_space_vertical, w_space_vertical), dtype=np.uint8) + 0x80

        def combine_l_pix_vertical(l_pix_part : List[np.ndarray]) -> np.ndarray:
            l = [l_pix_part[0]]
            for pix in l_pix_part[1:]:
                l.append(arr_space_vertical)
                l.append(pix)
            return np.vstack(l)
        
        pix_vertical = combine_l_pix_vertical(l_pix_horizontal)

        return pix_vertical

    # TODO: make this into separate functions too!
    pix_combine = combine_all_pix(l_pix=l_pix)
    file_path_combine = os.path.join(dir_path_combined_images, '{}.png'.format(folder_name))
    Image.fromarray(pix_combine).save(file_path_combine)

    l_pix_xor = [np.zeros(l_pix[0].shape, dtype=np.uint8) + 0x40] + [pix1 ^ pix2 for pix1, pix2 in zip(l_pix[:-1], l_pix[1:])]
    pix_combine_xor = combine_all_pix(l_pix=l_pix_xor)
    file_path_combine_xor = os.path.join(dir_path_combined_images_xor, '{}.png'.format(folder_name))
    Image.fromarray(pix_combine_xor).save(file_path_combine_xor)

    return dm_obj


if __name__ == '__main__':
    # with gzip.open('/run/user/1000/save_images/test_other_2021-06-30_12:51:47_E7FD686A/dm_obj.pkl.gz', 'rb') as f:
    #     dm_obj = dill.load(f)

    d_l_t_2d_cells = get_d_l_t_2d_cells()
    
    l_t_2d_cells = d_l_t_2d_cells[(3, 3)]

    # d_func_data = create_random_function_data(frame=1, frame_wrap=True)
    # # for _ in range(0, 10):
    # dm_obj = create_new_automaton(d_function_data=d_func_data, l_t_2d_cells=l_t_2d_cells)

    root, l_dir, l_file = next(os.walk(DIR_PATH_SAVE_IMAGES))

    d_root_to_dm = {}
    l_dir_test_other = list(filter(lambda x: 'test_other_' in x, l_dir))
    for dir_name in l_dir_test_other:
        print("dir_name: {}".format(dir_name))

        root_2, l_dir_2, l_file_2 = next(os.walk(os.path.join(root, dir_name)))
        assert len(l_dir_2) == 0
        assert 'test_functions_python.py' in l_file_2
        assert 'dm_obj.pkl.gz' in l_file_2

        dm = load_pkl_gz_obj(os.path.join(root_2, 'dm_obj.pkl.gz'))
        d_root_to_dm[dir_name] = dm

    sys.exit()

    def create_many_new_automaton(n: int, iter_same_func: int, l_t_2d_cells: Any) -> None:
        np.random.seed()
        for i in range(0, n):
            print("i: {}".format(i))
            d_func_data = create_random_function_data(frame=1, frame_wrap=True)
            # use the same function iter_same_func times
            for _ in range(0, iter_same_func):
                create_new_automaton(d_function_data=d_func_data, l_t_2d_cells=l_t_2d_cells)

    # create_many_new_automaton(n=1, iter_same_func=1, l_t_2d_cells=l_t_2d_cells)

    # sys.exit(0)

    is_create_new_multiprocessing = False
    # is_create_new_multiprocessing = True

    # mult_proc_mng = MultiprocessingManager(cpu_count=7)
    if is_create_new_multiprocessing:
        mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())

        print('Define new Function!')
        mult_proc_mng.define_new_func('func_create_many_new_automaton', create_many_new_automaton)

        print('Do the jobs!!')
        l_arguments = []
        l_ret = mult_proc_mng.do_new_jobs(
            ['func_create_many_new_automaton']*mult_proc_mng.worker_amount,
            [(5, 10, l_t_2d_cells)]*mult_proc_mng.worker_amount,
            # [(30,)] * mult_proc_mng.cpu_count,
        )
        print("len(l_ret): {}".format(len(l_ret)))

        del mult_proc_mng

    l_dir_path = []
    for root, dirs, _ in os.walk(os.path.join(TEMP_DIR, 'save_images')):
        for directory in dirs:
            if 'test_other_' in directory:
                l_dir_path.append(os.path.join(root, directory))
        break

    l_dir_path = sorted(l_dir_path)
    print('len(l_dir_path): {}'.format(len(l_dir_path)))
    l_dm_obj = []
    l_func_str = []
    l_t_t_var_inv = []
    d_t_t_var_inv_to_t_t_var_inv_idx = {}
    d_t_t_var_inv_idx_to_t_t_var_inv = {}
    d_idx_to_dir_path = {}
    # for idx, t_t_var_inv in enumerate(l_t_t_var_inv, 0):
    #     d_t_t_var_inv_to_t_t_var_inv_idx[t_t_var_inv] = idx
    #     d_t_t_var_inv_idx_to_t_t_var_inv[idx] = t_t_var_inv
    l_dir_path_len = len(l_dir_path)
    l_data = []
    for i, dir_path in enumerate(l_dir_path, 1):
    # for i, dir_path in enumerate(l_dir_path[:1], 1):
        idx = i - 1
        file_path = os.path.join(dir_path, 'dm_obj.pkl.gz')
        print('Loading file {:4}/{:4}: {}'.format(i, l_dir_path_len, file_path))

        with gzip.open(file_path, 'rb') as f:
            dm_obj = dill.load(f)

        l_dm_obj.append(dm_obj)
        l_arr_historic_ranges.append(dm_obj.arr_historic_ranges)
        l_func_str.append(dm_obj.func_str)

        t_t_var_inv = dm_obj.d_function_data.l_dm_local[0]['t_t_var_inv']
        l_t_t_var_inv.append(t_t_var_inv)

        d_t_t_var_inv_to_t_t_var_inv_idx[t_t_var_inv] = idx
        d_t_t_var_inv_idx_to_t_t_var_inv[idx] = t_t_var_inv
        d_idx_to_dir_path[idx] = dir_path

        arr_sum_pix_cluster_roll = dm_obj.arr_sum_pix_cluster_roll
        arr = arr_sum_pix_cluster_roll.astype(np.int64)
        arr_concat_1st_2nd_deriv = np.concatenate((arr[:-2], arr[1:-1]-arr[:-2], (arr[2:]-arr[1:-1])-(arr[1:-1]-arr[:-2])), axis=1)

        l_data.extend([(idx, idx_img, arr_row) for idx_img, arr_row in enumerate(arr_concat_1st_2nd_deriv, 0)])

    df = pd.DataFrame(data=l_data, columns=['idx', 'idx_img', 'arr_1st_2nd_deriv'], dtype=object)
    df['t_1st_2nd_deriv'] = pd.Series(data=[tuple(arr.tolist()) for arr in df['arr_1st_2nd_deriv'].values], index=df.index, dtype=object)

    arr_idx = df.duplicated(subset=['idx', 't_1st_2nd_deriv']).values
    df_drop = df.loc[~arr_idx].copy().sort_values(by=['t_1st_2nd_deriv', 'idx'])

    print("df.shape: {}".format(df.shape))
    print("df_drop.shape: {}".format(df_drop.shape))

    arr_idx = df_drop['idx'].values
    arr_arr_1st_2nd_deriv = np.array([a.astype(np.int64) for a in df_drop['arr_1st_2nd_deriv'].values])
    arr_t_1st_2nd_deriv = df_drop['t_1st_2nd_deriv'].values

    u, c = np.unique(arr_t_1st_2nd_deriv, return_counts=True)
    u2, c2 = np.unique(c, return_counts=True)
