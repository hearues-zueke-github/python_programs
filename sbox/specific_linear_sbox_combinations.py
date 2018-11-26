#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

import numpy as np

from copy import deepcopy

import sys
sys.path.append("../encryption")

import Utils

if __name__ == "__main__":
    n = 16 # amount of elements = modulo = number base

    x = np.arange(0, n, dtype=np.int)

    sbox_amount = 800000

    factors_lin = np.random.randint(0, n, (3, sbox_amount))
    
    factors_spec_lin = np.random.randint(0, n, (24, sbox_amount))
    # mods = [2, 4, 8, 15] # (2,2,4,4,8) best for (24, 5)
    mods = np.arange(2, n)
    mods_len = 5
    mods_spec_lin = np.sort(np.random.choffice(mods, (mods_len, )))

    print("mods_spec_lin: {}".format(mods_spec_lin))

    def apply_linear_factors(n, ass):
        x = np.arange(0, n, dtype=np.int).reshape((-1, 1))
        return (ass[0]+ass[1]*x).T % n
    
    def apply_specific_linear_factors(n, ass, bss):
        x = np.arange(0, n, dtype=np.int).reshape((-1, 1))
        expr_str_temp = "(ass[{}]+ass[{}]*x+ass[{}]*x**2+ass[{}]*x**3)"
        whole_str = "("
        for i in range(0, 6):
            whole_str += ("" if i == 0 else "+")+expr_str_temp.format(*np.arange(0, 4)+4*i)+"%"+("n" if i == 0 else "bss[{}]".format(i-1))
        whole_str += ").T %n"
        # print("whole_str: {}".format(whole_str))
        # sys.exit(-1)
        return eval(whole_str)
        # return ((ass[0]+ass[1]*x+ass[2]*x**2+ass[3]*x**3)%n+
        #         (ass[4]+ass[5]*x+ass[6]*x**2+ass[7]*x**3)%bss[0]+
        #         (ass[8]+ass[9]*x+ass[10]*x**2+ass[11]*x**3)%bss[1]).T % n
        # return ((ass[0]+ass[1]*x)%n+
        #         (ass[2]+ass[3]*x)%bss[0]+
        #         (ass[4]+ass[5]*x)%bss[1]+
        #         (ass[6]+ass[7]*x)%bss[2]+
        #         (ass[8]+ass[9]*x)%bss[3]+
        #         (ass[10]+ass[11]*x)%bss[4]).T % n
        # return ((ass[0]+ass[1]*x)%bss[4]+
        #         (ass[2]+ass[3]*x)%bss[0]+
        #         (ass[4]+ass[5]*x)%bss[1]+
        #         (ass[6]+ass[7]*x)%bss[2]+
        #         (ass[8]+ass[9]*x)%bss[3]).T % n

    def get_unique_sbox_only(fs_x):
        return fs_x[(lambda x: ~np.any(x[:, 1:]-x[:, :-1]!=1, axis=1))(np.sort(fs_x, axis=1))]

    # fs_x_lin = apply_linear_factors(n, factors_lin)
    # fs_x_lin_unique = get_unique_sbox_only(fs_x_lin)
    # print("fs_x_lin_unique.shape: {}".format(fs_x_lin_unique.shape))
    # fs_x_lin_def_unique = list(set(list(map(tuple, fs_x_lin_unique.tolist()))))
    # print("len(fs_x_lin_def_unique): {}".format(len(fs_x_lin_def_unique)))

    get_new_mods = lambda: np.sort(np.random.choice(mods, (mods_len, )))
    mods_spec_lin = get_new_mods()
    best_mods = mods_spec_lin.copy()
    best_unique_sboxes_len = 0

    while True:
        fs_x_spec_lin = apply_specific_linear_factors(n, factors_spec_lin, mods_spec_lin)
        fs_x_spec_lin_unique = get_unique_sbox_only(fs_x_spec_lin)
        print("fs_x_spec_lin_unique.shape: {}".format(fs_x_spec_lin_unique.shape))
        fs_x_spec_lin_def_unique = list(set(list(map(tuple, fs_x_spec_lin_unique.tolist()))))
        unique_sboxes_len = len(fs_x_spec_lin_def_unique)
        # print("unique_sboxes_len: {}".format(unique_sboxes_len))
        print("unique_sboxes_len: {}, mods_spec_lin: {}".format(unique_sboxes_len, mods_spec_lin))

        if best_unique_sboxes_len < unique_sboxes_len:
            best_unique_sboxes_len = unique_sboxes_len
            best_mods = mods_spec_lin.copy()

        print("best_unique_sboxes_len: {}, best_mods: {}".format(best_unique_sboxes_len, best_mods))
        # print("best_unique_sboxes_len: {}".format(best_unique_sboxes_len))
        # print("best_mods: {}".format(best_mods))

        mods_spec_lin = get_new_mods()
