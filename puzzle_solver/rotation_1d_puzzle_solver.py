#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from colorama import Fore, Style
from copy import deepcopy

path_root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"
print("path_root_dir: {}".format(path_root_dir))

sys.path.append(path_root_dir+"../combinatorics/")
import different_combinations

class Node(Exception):
    def __init__(self):
        self.parent = None
        self.childs = {}
        self.moves = None
        self.arr = None


def convert_num_to_base_tuple(n, b):
    assert b >= 2
    assert n >= 0

    tpl = ()
    # tpl = (n%b, )
    while n > 0:
        tpl += (n%b, )
        n //= b

    return tpl


if __name__ == "__main__":
    n = 9
    arr_1d = np.arange(0, n)
    multiply_mask = 8**np.arange(0, n)
    # print("multiply_mask: {}".format(multiply_mask))
    # sys.exit(-1)

    print("arr_1d: {}".format(arr_1d))

    # standard for rotating is 3 places/numbers!
    m_amount = 3
    idx = np.arange(0, m_amount)
    # rolls = [np.roll(idx, i) for i in range(1, m_amount)]

    rolls = different_combinations.get_permutation_table(m_amount, same_pos=False)
    print("rolls.shape: {}".format(rolls.shape))

    rotation_idxs = {}
    for j in range(0, n+1-m_amount):
        for i in range(0, len(rolls)):
            rotation_idxs[(j+1, i+1)] = (idx+j, rolls[i]+j)

    print("len(rotation_idxs): {}".format(len(rotation_idxs)))

    moves = list(rotation_idxs.keys())
    print("len(moves): {}".format(len(moves)))

    def rot(arr, move):
        arr = np.array(arr)
        idxs1, idxs2 = rotation_idxs[move]
        arr[idxs1] = arr[idxs2]
        return arr


    # root = Node()
    # moves_now = ()
    # root.moves = moves_now
    # root.arr = arr_1d.copy()

    # def create_tree(node):
    #     arr = node.arr
    #     moves_now = node.moves

    #     for move in moves:
    #         moves_new = moves_now+(move, )
    #         arr_new = rot(arr, move)
    #         tpl = tuple(arr_new.tolist())

    #         if not tpl in create_tree.used_arr_tpl:
    #             create_tree.used_arr_tpl.append(tpl)
    #             create_tree.used_moves.append(moves_new)
    #             node_new = Node()
    #             node_new.parent = node
    #             node.childs[move] = node_new
    #             node_new.moves = moves_new
    #             node_new.arr = arr_new

    #             create_tree(node_new)
    # create_tree.used_arr_tpl = [tuple(arr_1d.tolist())]
    # create_tree.used_moves = [moves_now]
    # create_tree(root)

    # # print("create_tree.used_arr_tpl:\n{}".format(create_tree.used_arr_tpl))
    # print("len(create_tree.used_arr_tpl):\n{}".format(len(create_tree.used_arr_tpl)))
    # print("create_tree.used_moves[-1]:\n{}".format(create_tree.used_moves[-1]))
    # print("len(create_tree.used_moves):\n{}".format(len(create_tree.used_moves)))

    # idx_vals, rol_vals = np.array(create_tree.used_moves[-1]).T

    # print("idx_vals:\n{}".format(idx_vals))
    # print("rol_vals:\n{}".format(rol_vals))


    arr_perm_table = different_combinations.get_permutation_table(n)
    lst_perm_table_tpls_num = np.sum(np.multiply.reduce((arr_perm_table, multiply_mask)), axis=1)

    idx_dict = {tpl_num: i for i, tpl_num in enumerate(lst_perm_table_tpls_num)}
    state_move_dict = {tpl_num: {} for i, tpl_num in enumerate(lst_perm_table_tpls_num)}

    length = arr_perm_table.shape[0]
    for idx, tpl_num in enumerate(lst_perm_table_tpls_num):
        if idx % (length//10) == 0:
            print("idx: {}".format(idx))
        i = idx_dict[tpl_num]
        row = arr_perm_table[i]

        state_move = state_move_dict[tpl_num]
        for move in moves:
            arr_new = rot(row, move)
            tpl_num_new = np.sum(arr_new*multiply_mask)
            state_move[move] = (tpl_num_new, tuple(arr_new.tolist()))


    # tpls_num_now = [np.sum(arr_1d.astype(np.uint64)*multiply_mask)]
    # tpls_now = [tuple((arr_1d).tolist())]
    tpls_num_now = [np.sum(arr_1d*multiply_mask)]
    # tpls_num_now = [(np.sum(arr_1d.astype(np.uint64)*multiply_mask), tuple((arr_1d).tolist()))]

    used_tpls_num = {tpls_num_now[0]: 0}
    # used_tpls_num = [tpls_num_now[0]]
    steps_tpls_dict = {0: [tuple(arr_1d.tolist())]}
    tpls_num_moves_dict = {tpls_num_now[0]: ()}
    # tpls_num_moves_dict = {tpls_num_now[0]: ()}

    steps = 1
    while len(tpls_num_now) > 0:
        tpls_num_tpls_next = []

        for tpl_num in tpls_num_now:
        # for tpl_num, tpl in tpls_num_now:
            moves_now = tpls_num_moves_dict[tpl_num]
            state_move = state_move_dict[tpl_num]
            for move in moves:
                tpl_num_next, tpl_next = state_move[move]
                if tpl_num_next in used_tpls_num:
                    continue
                
                used_tpls_num[tpl_num_next] = 0
                # used_tpls_num.append(tpl_num_next)
                tpls_num_tpls_next.append(tpl_num_next)
                # tpls_num_tpls_next.append((tpl_num_next, tpl_next))

                moves_next = moves_now+(move, )
                tpls_num_moves_dict[tpl_num_next] = moves_next

        if len(tpls_num_tpls_next) == 0:
            break

        print("steps: {}, len(tpls_num_tpls_next): {}".format(steps, len(tpls_num_tpls_next)))

        steps_tpls_dict[steps] = tpls_num_tpls_next
        steps += 1

        tpls_num_now = tpls_num_tpls_next
        # tpls_now = tpls_num_tpls_next

    keys_steps = sorted(list(steps_tpls_dict.keys()))
    lens_amount = {step: len(steps_tpls_dict[step]) for step in keys_steps}

    print("lens_amount: {}".format(lens_amount))
