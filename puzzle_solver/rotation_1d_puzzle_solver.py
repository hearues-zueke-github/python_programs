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
    argv = sys.argv

    if len(argv) < 3:
        print("Needed 2 arguments!")
        print("usage: ./<program> <n> <m_amount>")
        sys.exit(-1)

    n_str = argv[1]
    m_amount_str = argv[2]
    if not n_str.isdigit():
        print("value 'n' is not a string!")
    if not m_amount_str.isdigit():
        print("value 'm_amount' is not a string!")

    n = int(n_str)
    m_amount = int(m_amount_str)

    # n = 4
    arr_1d = np.arange(0, n)
    multiply_mask = n**np.arange(0, n)
    # print("multiply_mask: {}".format(multiply_mask))
    # sys.exit(-1)

    print("arr_1d: {}".format(arr_1d))

    # standard for rotating is 3 places/numbers!
    # m_amount = 2
    idx = np.arange(0, m_amount)
    # rolls = [np.roll(idx, i) for i in range(1, m_amount)]

    rolls = different_combinations.get_permutation_table(m_amount, same_pos=False)
    print("rolls.shape: {}".format(rolls.shape))

    rotation_idxs = []
    # rotation_idxs = {}
    for j in range(0, n+1-m_amount):
        for i in range(0, len(rolls)):
            rotation_idxs.append((j+1, i+1))

    print("len(rotation_idxs): {}".format(len(rotation_idxs)))

    # moves = list(rotation_idxs.keys())
    # moves = list(rotation_idxs.keys())
    moves = list(rotation_idxs)
    print("len(moves): {}".format(len(moves)))

    # def rot(arr, move):
    #     arr = np.array(arr)
    #     idxs1, idxs2 = rotation_idxs[move]
    #     arr[idxs1] = arr[idxs2]
    #     return arr


    rolls_flat = rolls.reshape((-1, ))
    print("rolls_flat: {}".format(rolls_flat))

    rows = n-m_amount+1
    ys_2d = np.zeros((rows, rolls_flat.shape[0]), dtype=np.int).reshape((-1, m_amount))
    ys_2d += np.arange(0, ys_2d.shape[0]).reshape((-1, 1))
    # ys_2d = np.zeros(((n-m_amount+1), m_amount*rolls.shape[0]), dtype=np.int)+np.arange(0, n-m_amount+1).reshape((-1, 1))

    xs_2d = np.zeros((rows, rolls_flat.shape[0]), dtype=np.int)+rolls_flat
    xs_2d += np.arange(0, rows).reshape((-1, 1))
    xs_2d = xs_2d.reshape((-1, m_amount))

    xs_orig_2d = (xs_2d*0+np.arange(0, m_amount)).reshape((-1, ))

    print("ys_2d:\n{}".format(ys_2d))
    print("xs_2d:\n{}".format(xs_2d))
    print("xs_orig_2d:\n{}".format(xs_orig_2d))

    ys = ys_2d.reshape((-1, ))
    xs = xs_2d.reshape((-1, ))
    xs_orig = (xs_orig_2d.reshape((rows, -1))+np.arange(0, rows).reshape((-1, 1))).reshape((-1, ))

    ps_orig = (ys, xs_orig)
    ps = (ys, xs)

    print("ps_orig:\n{}".format(ps_orig))
    print("ps:\n{}".format(ps))


    def get_state_move(arr_1d):
        arrs = np.zeros((ys_2d.shape[0], arr_1d.shape[0]), dtype=np.int)+arr_1d
        arrs[ps_orig] = arrs[ps]
        rows_num = np.sum(arrs*multiply_mask, axis=1)
        
        state_move = {}
        for move, row_num, row in zip(moves, rows_num, arrs):
            state_move[move] = (row_num, row)

        return state_move

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
    lst_perm_table_rows_num = np.sum(np.multiply.reduce((arr_perm_table, multiply_mask)), axis=1)

    idx_dict = {row_num: i for i, row_num in enumerate(lst_perm_table_rows_num)}
    # state_move_dict = {row_num: {} for i, row_num in enumerate(lst_perm_table_rows_num)}

    # length = arr_perm_table.shape[0]
    # for idx, row_num in enumerate(lst_perm_table_rows_num):
    #     if idx % (length//10) == 0:
    #         print("idx: {}".format(idx))
    #     i = idx_dict[row_num]
    #     row = arr_perm_table[i]

    #     state_move_dict[row_num] = get_state_move(row)

    # rows_num_now = [np.sum(arr_1d.astype(np.uint64)*multiply_mask)]
    # tpls_now = [tuple((arr_1d).tolist())]
    rows_num_now = [np.sum(arr_1d*multiply_mask)]
    # rows_num_now = [(np.sum(arr_1d.astype(np.uint64)*multiply_mask), tuple((arr_1d).tolist()))]

    used_tpls_num = {rows_num_now[0]: 0}
    # used_tpls_num = [rows_num_now[0]]
    steps_rows_dict = {0: [arr_1d]}
    rows_num_moves_dict = {rows_num_now[0]: ()}
    # rows_num_moves_dict = {rows_num_now[0]: ()}

    steps = 1
    while len(rows_num_now) > 0:
        rows_num_next = []

        for row_num in rows_num_now:
            idx = idx_dict[row_num]
            row = arr_perm_table[idx]


        # for row_num, tpl in rows_num_now:
            moves_now = rows_num_moves_dict[row_num]
            state_move = get_state_move(row)
            # state_move = state_move_dict[row_num]
            for move in moves:
                tpl_num_next, tpl_next = state_move[move]
                if tpl_num_next in used_tpls_num:
                    continue
                
                used_tpls_num[tpl_num_next] = 0
                # used_tpls_num.append(tpl_num_next)
                rows_num_next.append(tpl_num_next)
                # rows_num_next.append((tpl_num_next, tpl_next))

                moves_next = moves_now+(move, )
                rows_num_moves_dict[tpl_num_next] = moves_next

        if len(rows_num_next) == 0:
            break

        print("steps: {}, len(rows_num_next): {}".format(steps, len(rows_num_next)))

        steps_rows_dict[steps] = rows_num_next
        steps += 1

        rows_num_now = rows_num_next
        # tpls_now = rows_num_next

    keys_steps = sorted(list(steps_rows_dict.keys()))
    lens_amount = {step: len(steps_rows_dict[step]) for step in keys_steps}
    print("lens_amount: {}".format(lens_amount))

    path_figures = path_root_dir+"figures/"
    if not os.path.exists(path_figures):
        os.makedirs(path_figures)

    y_lens = list(lens_amount.values())
    x_lens = np.arange(0, len(y_lens))

    plt.figure()

    plt.title("1d-puzzle:\nn: {}, m_amount: {}".format(n, m_amount))
    plt.xlabel("Steps needed")
    plt.ylabel("Number of steps")

    plt.plot(x_lens, y_lens, "b-o")

    # plt.show()
    plt.savefig(path_figures+"n_{}_m_amount_{}.png".format(n, m_amount))
