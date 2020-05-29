#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import sys

import matplotlib.pyplot as plt

import numpy as np

from copy import deepcopy
from math import factorial as fac

sys.path.append("../combinatorics/")
from different_combinations import get_all_combinations_repeat, get_permutation_table

from find_graph_cycles import get_cycles_of_1_directed_graph

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

def write_digraph_as_dotfile(path, l_x, l_y):
    with open(path, 'w') as f:
        f.write('digraph {\n')
        d = {x: i for i, x in enumerate(l_x, 0)}
        for x in l_x:
            f.write(f'  x{d[x]}[label="{x}"];\n')
        f.write('\n')
        for x, y in zip(l_x, l_y):
            f.write(f'  x{d[x]} -> x{d[y]};\n')

        f.write('}\n')


def convert_list_tuples_to_nums(l_tpls):
    d_convert = {t: i for i, t in enumerate(l_tpls, 0)}
    d_convert_inv = {i: t for t, i in d_convert.items()}
    return d_convert, d_convert_inv


def restore_sbox(sbox, sbox_prev):
    sbox = sbox.copy()

    idxs = sbox==np.arange(0, sbox.shape[0])
    idxs_ident = np.where(idxs)[0]
    if idxs_ident.shape[0]==0:
        return sbox

    if idxs_ident.shape[0]==1:
        idx = idxs_ident[0]
        for j in range(0, n):
            k = sbox_prev[j]
            if k!=idx:
                sbox[j], sbox[k] = sbox[k], sbox[j]
                return sbox

    len_idxs_ident = idxs_ident.shape[0]
    idxs_ident_ident = np.arange(0, len_idxs_ident)
    is_found_roll = False
    for j in range(0, n-len_idxs_ident+1):
        idxs_roll = np.argsort(sbox_prev[j:j+len_idxs_ident])
        if np.any(idxs_roll==idxs_ident_ident):
            continue

        sbox[idxs_ident] = sbox[idxs_ident[idxs_roll]]
        return sbox

    sbox1_prev_part = np.hstack((sbox_prev[-len_idxs_ident+1:], sbox_prev[:len_idxs_ident-1]))
    for j in range(0, len_idxs_ident-1):
        idxs_roll = np.argsort(sbox1_prev_part[j:j+len_idxs_ident])
        if np.any(idxs_roll==idxs_ident_ident):
            continue

        sbox[idxs_ident] = sbox[idxs_ident[idxs_roll]]
        return sbox

    sbox[idxs_ident] = sbox[np.roll(idxs_ident, 1)]
    return sbox


if __name__=='__main__':
    n = 5
    sbox_ident = np.arange(0, n)
    perm_tbl = get_permutation_table(n, is_same_pos=False)
    
    s_t_cycles = set()
    s_t_cycles_tpl = set()
    s_t_no_cylces = set()
    for perm in perm_tbl:
        for perm_prev in perm_tbl:
            sbox1 = perm_prev.copy()
            sbox2 = perm.copy()

            t = (tuple(sbox1.tolist()), tuple(sbox2.tolist()))
            if t in s_t_no_cylces or t in s_t_cycles:
                continue
            l_perms = [t]
            s_perms = set()
            is_finished_earlier = False
            while True:
                sbox1_new = np.argsort((np.cumsum(sbox2)+sbox1[sbox2])%n)
                sbox2_new = np.argsort((np.cumsum(sbox2)+sbox2[sbox1]+sbox1_new)%n)

                sbox1 = restore_sbox(sbox1_new, sbox2)
                sbox2 = restore_sbox(sbox2_new, sbox1)

                t = (tuple(sbox1.tolist()), tuple(sbox2.tolist()))
                if t in s_t_no_cylces or t in s_t_cycles:
                    is_finished_earlier = True
                    break
                if t in s_perms:
                    l_perms.append(t)
                    break
                l_perms.append(t)
                s_perms.add(t)

            if is_finished_earlier:
                if len(l_perms)>1:
                    if l_perms[-1] in s_t_cycles:
                        s_t_no_cylces.update(l_perms[:-1])
                    else:
                        s_t_no_cylces.update(l_perms)
                continue

            idx_start = l_perms.index(l_perms[-1])
            l_perms_cycle = l_perms[idx_start:-1]
            idx_start_cycle = l_perms_cycle.index(sorted(l_perms_cycle)[0])
            l_perms_cycle_start = l_perms_cycle[idx_start_cycle:]+l_perms_cycle[:idx_start_cycle]
            s_t_cycles_tpl.add(tuple(l_perms_cycle_start))
            s_t_cycles.update(l_perms_cycle_start)
            s_t_no_cylces.update(l_perms[:idx_start])
            # print("n: {}, perm: {}, l_perms: {}".format(n, perm, l_perms))

    # print("s_t_cycles_tpl: {}".format(s_t_cycles_tpl))
    l_t_cycles_tpl = sorted(s_t_cycles_tpl, key=lambda x: (len(x), x))
    for i, t_cycles_tpl in enumerate(l_t_cycles_tpl, 0):
        print("i: {}, len(t_cycles_tpl): {}".format(i, len(t_cycles_tpl)))

    unique_t_in_l_t_cycles_tpl = len(set(list(map(tuple, np.array(l_t_cycles_tpl[-1]).reshape((-1, n))))))
    print("unique_t_in_l_t_cycles_tpl: {}".format(unique_t_in_l_t_cycles_tpl))

    sys.exit(0)
    
    n = 10
    perm_tbl = get_permutation_table(n)
    next_perm_tbl = np.argsort(np.cumsum(perm_tbl, axis=1)%n, axis=1)

    l_x = [tuple(perm) for perm in perm_tbl]
    l_y = [tuple(next_perm) for next_perm in next_perm_tbl]

    d_convert, d_convert_inv = convert_list_tuples_to_nums(l_x)

    l_x_conv = list(map(lambda x: d_convert[x], l_x))
    l_y_conv = list(map(lambda x: d_convert[x], l_y))

    # d = {x: y for x, y in zip(l_x, l_y)}
    l = [(x, y) for x, y in zip(l_x_conv, l_y_conv)]
    # l = [(x, y) for x, y in zip(l_x, l_y)]

    l_cycles_nums = get_cycles_of_1_directed_graph(l)
    l_cycles = [[d_convert_inv[x] for x in l_cycle_num] for l_cycle_num in l_cycles_nums]
    l_cycles_lens = list(map(len, l_cycles_nums))

    print("l_cycles: {}".format(l_cycles))
    print("l_cycles_nums: {}".format(l_cycles_nums))
    print("l_cycles_lens: {}".format(l_cycles_lens))

    write_digraph_as_dotfile(PATH_ROOT_DIR+f'sbox_cumsum_graph_n_{n}.dot', l_x, l_y)
