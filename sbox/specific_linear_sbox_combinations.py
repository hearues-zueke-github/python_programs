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
    
    factors_spec_lin = np.random.randint(0, n, (10, sbox_amount))
    mods = [2, 4, 8, 15]
    mods_spec_lin = np.random.choice(mods, (4, ))
    # mods_spec_lin = np.random.randint(2, n, (4, ))

    print("mods_spec_lin: {}".format(mods_spec_lin))

    def apply_linear_factors(n, ass):
        x = np.arange(0, n, dtype=np.int).reshape((-1, 1))
        return (ass[0]+ass[1]*x).T % n
        # return (ass[0]+ass[1]*x+(x+ass[2])%3).T % n
    
    def apply_specific_linear_factors(n, ass, bss):
        x = np.arange(0, n, dtype=np.int).reshape((-1, 1))
        # return (ass[0]+ass[1]*x+ass[2]*x**2).T % n
        # return (ass[0]+ass[1]*x+ass[2]*x**2+(ass[3]+x)%2).T % n
        # return (ass[0]+ass[1]*x+ass[2]*x**2+(ass[3]+ass[4]*x+ass[5]*x**2)%4).T % n
        # return (ass[0]+ass[1]*x+ass[2]*x**2+(ass[3]+ass[4]*x)%2+(ass[5]+ass[6]*x)%2).T % n
        # return (ass[0]+ass[1]*x+ass[2]*x**2+(ass[3]+ass[4]*x)%2+(ass[5]+ass[6]*x)%2+(ass[7]+ass[8]*x)%2).T % n
        return ((ass[0]+ass[1]*x)+
                (ass[2]+ass[3]*x)%bss[0]+# (n//2)+
                (ass[4]+ass[5]*x)%bss[1]+# (n//2)+
                (ass[6]+ass[7]*x)%bss[2]+# (n//4)+
                (ass[8]+ass[9]*x)%bss[3]).T % n# (n//4)).T % n

    def get_unique_sbox_only(fs_x):
        return fs_x[(lambda x: ~np.any(x[:, 1:]-x[:, :-1]!=1, axis=1))(np.sort(fs_x, axis=1))]

    # fs_x_lin = apply_linear_factors(n, factors_lin)
    # fs_x_lin_unique = get_unique_sbox_only(fs_x_lin)
    # print("fs_x_lin_unique.shape: {}".format(fs_x_lin_unique.shape))
    # fs_x_lin_def_unique = list(set(list(map(tuple, fs_x_lin_unique.tolist()))))
    # print("len(fs_x_lin_def_unique): {}".format(len(fs_x_lin_def_unique)))

    fs_x_spec_lin = apply_specific_linear_factors(n, factors_spec_lin, mods_spec_lin)
    fs_x_spec_lin_unique = get_unique_sbox_only(fs_x_spec_lin)
    print("fs_x_spec_lin_unique.shape: {}".format(fs_x_spec_lin_unique.shape))
    fs_x_spec_lin_def_unique = list(set(list(map(tuple, fs_x_spec_lin_unique.tolist()))))
    print("len(fs_x_spec_lin_def_unique): {}".format(len(fs_x_spec_lin_def_unique)))
