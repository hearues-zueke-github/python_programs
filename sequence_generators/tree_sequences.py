#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import sys
import time

import numpy as np

from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

import utils_sequence

class Tree(Exception):
    def __init__(self):
        self.p = None
        self.l = None
        self.r = None
        self.steps = -1
        self.num = -1


class TreeNodeMany(Exception):
    def __init__(self):
        self.p = None
        self.c = []
        # self.l = None
        # self.r = None
        self.steps = -1
        self.num = -1


def get_numbers_by_steps(root):
    def go_tree(node):
        steps = node.steps

        if not steps in go_tree.steps_dict_lst:
            go_tree.steps_dict_lst[steps] = []
        go_tree.steps_dict_lst[steps].append(node.num)

        for n in node.c:
            go_tree(n)
        # if node.l != None:
        #     go_tree(node.l)
        # if node.r != None:
        #     go_tree(node.r)
    go_tree.steps_dict_lst = {}

    go_tree(root)

    return go_tree.steps_dict_lst


if __name__ == '__main__':
    print('Hello World!')

    n = 300
    nodes = {i: TreeNodeMany() for i in range(1, n+1)}

    root = nodes[1]
    root.steps = 1
    root.num = 1

    next_num_dict = {1: 0}
    steps_dict = {1: 1}

    # div_num = 2
    for n in range(2, n+1):
        t1 = nodes[n]
        
        if (n-1) % 3 == 0:
        # if n % 2 == 0:
            n_next = (n-1)//3
            # n_next = n//2
            t2 = nodes[n_next]
            t2.c.append(t1)
            # t2.r = t1
        elif n % 3 == 0:
            n_next = n//3
            t2 = nodes[n_next]
            t2.c.append(t1)
        # elif n % 5 == 0:
        #     n_next = n//5
        #     t2 = nodes[n_next]
        #     t2.c.append(t1)
        else:
            n_next = n-1
            t2 = nodes[n_next]
            t2.l = t1
        
        next_num_dict[n] = n_next
        steps_dict[n] = steps_dict[n_next]+1
        
        t1.p = t2
        t1.steps = steps_dict[n]
        t1.num = n

    lst = list(zip(next_num_dict.keys(), next_num_dict.values(), steps_dict.values()))
    print("lst:\n{}".format(lst))

    steps_dict_lst = get_numbers_by_steps(root)

    print("steps_dict_lst:\n{}".format(steps_dict_lst))

    keys = sorted(list(steps_dict_lst.keys()))
    print("keys: {}".format(keys))

    concat_lst = []
    for k in keys:
        concat_lst += steps_dict_lst[k]

    print("concat_lst: \n{}".format(concat_lst))
