#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import random
import sys

from copy import deepcopy
from dotmap import DotMap

from sortedcontainers import SortedSet

from collections import defaultdict

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from math import gcd
from time import time
from functools import reduce

from PIL import Image

import numpy as np

import itertools

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style


def time_measure(f, args):
    start_time = time()
    ret = f(*args)
    end_time = time()
    diff_time = end_time-start_time
    return ret, diff_time


class ShowImg(Frame, object):
    def __init__(self, img):
        parent = Tk()
        Frame.__init__(self, parent)
        self.pack(fill=BOTH, expand=1)
        label1 = Label(self)
        label1.photo= ImageTk.PhotoImage(img)
        label1.config(image=label1.photo)
        label1.pack(fill=BOTH, expand=1)
        parent.mainloop()


def get_bounces_of_a_billard_table(w, h):
    x = 0 # starting x
    y = 0 # starting y

    end_positions = [(0, 0), (0, h), (w, 0), (w, h)]

    directions = {'UR': (1, 1), 'UL': (1, -1), 'DR': (-1, 1), 'DL': (-1, -1)}
    change_direction = {
        ('UR', 'R'): 'UL', ('UR', 'U'): 'DR',
        ('DR', 'R'): 'DL', ('DR', 'D'): 'UR',
        ('UL', 'L'): 'UR', ('UL', 'U'): 'DL',
        ('DL', 'L'): 'DR', ('DL', 'D'): 'UL',
    }
    bouncing_x = {0: 'L', w: 'R'}
    bouncing_y = {0: 'D', h: 'U'}

    direction = 'UR'
    dy, dx = directions[direction]
    amount_bounces = 0
    bounces_lst = []
    while True:
        x += dx
        y += dy

        # print("x: {}, y: {}".format(x, y))

        if (x, y) in end_positions:
            break

        if x in bouncing_x:
            # print("bouncing_x")
            bounce_side = bouncing_x[x]
            # print("bounce_side: {}".format(bounce_side))
            bounces_lst.append(bounce_side)
            direction = change_direction[(direction, bounce_side)]
            # print("direction: {}".format(direction))
            dy, dx = directions[direction]
            # print("dy: {}, dx: {}".format(dy, dx))
            amount_bounces += 1
        elif y in bouncing_y:
            # print("bouncing_y")
            bounce_side = bouncing_y[y]
            # print("bounce_side: {}".format(bounce_side))
            bounces_lst.append(bounce_side)
            direction = change_direction[(direction, bounce_side)]
            # print("direction: {}".format(direction))
            dy, dx = directions[direction]
            # print("dy: {}, dx: {}".format(dy, dx))
            amount_bounces += 1

    # print("amount_bounces: {}".format(amount_bounces))
    # print("bounces_lst: {}".format(bounces_lst))

    return amount_bounces, bounces_lst


def get_bounces_amount_of_a_billard_table(h, w):
    x = 0 # starting x
    y = 0 # starting y
    amount_bounces = 0; state = 0; do_loop = True
    lst = [(0, 0)]
    while do_loop:
        if state==0: # UR
            while x<w and y<h: x+=1;y+=1;lst.append((x, y))
            if x==w and y<h: amount_bounces+=1;state=2
            elif x<w and y==h: amount_bounces+=1;state=1
            else: do_loop = False
        elif state==1: # DR
            while x<w and y>0: x+=1;y-=1;lst.append((x, y))
            if x==w and y>0: amount_bounces+=1;state=3
            elif x<w and y==0: amount_bounces+=1;state=0
            else: do_loop = False
        elif state==2: # UL
            while x>0 and y<h: x-=1;y+=1;lst.append((x, y))
            if x==0 and y<h: amount_bounces+=1;state=0
            elif x>0 and y==h: amount_bounces+=1;state=3
            else: do_loop = False
        elif state==3: # DL
            while x>0 and y>0: x-=1;y-=1;lst.append((x, y))
            if x==0 and y>0: amount_bounces+=1;state=1
            elif x>0 and y==0: amount_bounces+=1;state=2
            else: do_loop = False
    # print("h: {}, w: {}".format(h, w))
    # print("lst: {}".format(lst))
    return amount_bounces

# simple testcases for different billard table sizes!
amount_bounces, bounces_lst = get_bounces_of_a_billard_table(1, 1)
assert amount_bounces==0
assert bounces_lst==[]

amount_bounces, bounces_lst = get_bounces_of_a_billard_table(5, 5)
assert amount_bounces==0
assert bounces_lst==[]

amount_bounces, bounces_lst = get_bounces_of_a_billard_table(3, 2)
assert amount_bounces==3
assert bounces_lst==['U', 'R', 'D']

amount_bounces, bounces_lst = get_bounces_of_a_billard_table(4, 3)
assert amount_bounces==5
assert bounces_lst==['U', 'R', 'D', 'L', 'U']

amount_bounces, bounces_lst = get_bounces_of_a_billard_table(5, 3)
assert amount_bounces==6
assert bounces_lst==['U', 'R', 'D', 'U', 'L', 'D']


assert get_bounces_amount_of_a_billard_table(1, 1)==0
assert get_bounces_amount_of_a_billard_table(4, 4)==0
assert get_bounces_amount_of_a_billard_table(3, 2)==3
assert get_bounces_amount_of_a_billard_table(4, 3)==5
assert get_bounces_amount_of_a_billard_table(5, 3)==6

for w in range(1, 11):
    for h in range(w+1, 11):
        assert get_bounces_of_a_billard_table(w, h)[0]==get_bounces_amount_of_a_billard_table(w, h)

# only 2d so far!
def calc_billard_all_possible_ball_moves_2d(size1, size2):
    s_corner_pos = set([(i, j) for i in [0, size1] for j in [0, size2]])
    s_edge_pos = set(
        [(i, 0) for i in range(1, size1)]+
        [(i, size2) for i in range(1, size1)]+
        [(0, i) for i in range(1, size2)]+
        [(size1, i) for i in range(1, size2)]
    )

    s_corner_edge_pos = s_corner_pos|s_edge_pos
    s_rest_edge = deepcopy(s_corner_edge_pos)

    l_corner_pos = sorted(list(s_corner_pos))
    l_edge_pos = sorted(list(s_edge_pos))

    def get_list_of_pos(p1, p2, check_set):
        l_pos = [(p1, p2)]
        l_pos_edge = [(p1, p2)]

        if p1==size1:
            add1 = -1
        else:
            add1 = 1

        if p2==size2:
            add2 = -1
        else:
            add2 = 1

        while True:
            is_edge = False
            if add1==+1 and p1>=size1:
                add1 = -1
                if not is_edge:
                    is_edge = True
                    l_pos_edge.append((p1, p2))
            elif add1==-1 and p1<=0:
                add1 = +1
                if not is_edge:
                    is_edge = True
                    l_pos_edge.append((p1, p2))

            if add2==+1 and p2>=size2:
                add2 = -1
                if not is_edge:
                    is_edge = True
                    l_pos_edge.append((p1, p2))
            elif add2==-1 and p2<=0:
                add2 = +1
                if not is_edge:
                    is_edge = True
                    l_pos_edge.append((p1, p2))

            p1 += add1
            p2 += add2

            l_pos.append((p1, p2))
            if (p1, p2) in check_set:
                break
        l_pos_edge.append((p1, p2))

        return l_pos, l_pos_edge


    l_pos_list = []
    l_pos_edge_list = []

    while len(l_corner_pos)>0:
        p1, p2 = l_corner_pos.pop(0)
        l_pos, l_pos_edge = get_list_of_pos(p1, p2, check_set=s_corner_pos)

        t = l_pos[-1]
        if t in l_corner_pos:
            l_corner_pos.remove(t)

        l_pos_list.append(l_pos)
        l_pos_edge_list.append(l_pos_edge)
        s_rest_edge = s_rest_edge-set(l_pos_edge)

    while len(s_rest_edge)>0:
        l_rest_edge = sorted(list(s_rest_edge))

        p1, p2 = l_rest_edge.pop(0)
        l_pos, l_pos_edge = get_list_of_pos(p1, p2, check_set=set([(p1, p2)]))

        l_pos_list.append(l_pos)
        l_pos_edge_list.append(l_pos_edge)
        s_rest_edge = s_rest_edge-set(l_pos_edge)

    return l_pos_list, l_pos_edge_list


def calc_billard_all_possible_ball_moves_n_dim(t_size):
    dimension = len(t_size)
    product = itertools.product

    l_01 = list(itertools.product(*[[0, 1] for _ in range(0, len(t_size))]))
    arr_01 = np.array(l_01)
    l_amount_01 = np.sum(arr_01, axis=1).tolist()

    l_01_lists = [[] for i in range(0, len(t_size)+1)]
    for l, amount in zip(l_01, l_amount_01):
        l_01_lists[amount].append(l)

    l_01_lists = l_01_lists[:-1]

    l_sets = [set(itertools.chain(*[itertools.product(*[[0, v] if i==0 else range(1, v) for i, v in  zip(l, t_size)]) for l in l_01])) for l_01 in l_01_lists]
    # globals()['l_sets'] = l_sets
    # print("l_sets: {}".format(l_sets))
    
    d_sets_directions_out = {v: list(itertools.product(*[[-1] if i==j else [1] if i==0 else [-1, 1] for i, j in zip(v, t_size)])) for l in l_sets for v in l}
    
    s_corner_pos = l_sets[0]
    s_corner_pos_add = set([(tp, ta) for tp in s_corner_pos for ta in d_sets_directions_out[tp]])
    l_corner_pos_add = sorted(list(s_corner_pos_add))

    l_edge_pos = sorted(itertools.chain(*[list(l) for l in l_sets[1:]]))
    s_edge_pos_add = {(tp, ta) for tp in l_edge_pos for ta in d_sets_directions_out[tp]}

    s_corner_edge_pos_add = s_corner_pos_add|s_edge_pos_add
    s_rest_edge_add = deepcopy(s_corner_edge_pos_add)

    # check sizes
    arr_size = np.array(t_size, dtype=object)
    n1 = np.multiply.reduce(arr_size+1)
    n2 = np.multiply.reduce(arr_size-1)
    length_points = sum([len(l) for l in l_sets])

    assert n1-n2==length_points
    assert len(s_corner_pos_add)==2**len(t_size)

    def get_list_of_pos(t_vals, t_add, check_set):
        l_pos = [(t_vals, t_add)]
        l_pos_edge = [(t_vals, t_add)]

        lp = list(t_vals)
        la = list(t_add)

        while True:
            is_edge = False

            for i in range(0, dimension):
                lp[i] += la[i]

            t_inv = (tuple(lp), tuple(-1 if i==1 else 1 for i in la))

            for i in range(0, dimension):
                a = la[i]
                p = lp[i]

                if a==+1 and p>=t_size[i]:
                    a = -1
                    if not is_edge:
                        is_edge = True
                elif a==-1 and p<=0:
                    a = +1
                    if not is_edge:
                        is_edge = True

                la[i] = a

            t = (tuple(lp), tuple(la))
            l_pos.append(t)
            if is_edge:
                if t_inv in s_rest_edge_add:
                    s_rest_edge_add.remove(t_inv)
                l_pos_edge.append(t)

            if t in check_set:
                break

        return l_pos, l_pos_edge


    l_pos_list = []
    l_pos_edge_list = []

    while len(l_corner_pos_add)>0:
        t_first = l_corner_pos_add[0]
        l_corner_pos_add.remove(t_first)

        tp_first, ta_first = t_first
        l_pos, l_pos_edge = get_list_of_pos(tp_first, ta_first, check_set=s_corner_pos_add)

        t_last = l_pos[-1]
        assert t_last in l_corner_pos_add
        l_corner_pos_add.remove(t_last)

        l_pos_list.append(l_pos)
        l_pos_edge_list.append(l_pos_edge)

        s_rest_edge_add = s_rest_edge_add-set(l_pos_edge)

    while len(s_rest_edge_add)>0:
        for t_first in s_rest_edge_add:
            break

        tp_first, ta_first = t_first
        l_pos, l_pos_edge = get_list_of_pos(tp_first, ta_first, check_set=set([t_first]))

        l_pos_list.append(l_pos)
        l_pos_edge_list.append(l_pos_edge)

        s_rest_edge_add = s_rest_edge_add-set(l_pos_edge)

    assert len(s_rest_edge_add)==0

    return l_pos_list, l_pos_edge_list


def do_testcases_for_2d_and_n_dim_functions():
    print('Executing testcases for 2d and n-dim functions.')
    # do some testcases for testing the billard moves of n-dim (2d) and 2d only!
    def rotate_list_to_most_minimum_value(l):
        l_sort = sorted(l)
        min_val = l_sort[0]
        l_min_idxs = [i for i, v in enumerate(l, 0) if v==min_val]

        if len(l_min_idxs)==1:
            i = l_min_idxs[0]
            return list(l[i:]+l[:i])

        return sorted([list(l[i:]+l[:i]) for i in l_min_idxs])[0]

    l_t_size = [(i, j) for i in range(1, 21) for j in range(1, 21)]
    for size1, size2 in l_t_size:
        # print("Checking for: size1: {}, size2: {}".format(size1, size2))
        l_pos_list_1_full, l_pos_edge_list_1_full = calc_billard_all_possible_ball_moves_2d(size1, size2)
        l_pos_list_1 = [l if l[0]!=l[-1] else l[:-1] for l in l_pos_list_1_full]
        l_pos_edge_list_1 = [l if l[0]!=l[-1] else l[:-1] for l in l_pos_edge_list_1_full]

        l_pos_list, l_pos_edge_list = calc_billard_all_possible_ball_moves_n_dim([size1, size2])
        l_pos_list_2 = [[t for t, _ in (l if l[0]!=l[-1] else l[:-1])] for l in l_pos_list]
        l_pos_edge_list_2 = [[t for t, _ in (l if l[0]!=l[-1] else l[:-1])] for l in l_pos_edge_list]


        l_pos_list_1_sorted = sorted([rotate_list_to_most_minimum_value(l) for l in l_pos_list_1])
        l_pos_edge_list_1_sorted = sorted([rotate_list_to_most_minimum_value(l) for l in l_pos_edge_list_1])
        
        l_pos_list_2_sorted = sorted([rotate_list_to_most_minimum_value(l) for l in l_pos_list_2])
        l_pos_edge_list_2_sorted = sorted([rotate_list_to_most_minimum_value(l) for l in l_pos_edge_list_2])
        
        l_pos_list_2_rot_sorted = sorted([rotate_list_to_most_minimum_value(l[::-1]) for l in l_pos_list_2])
        l_pos_edge_list_2_rot_sorted = sorted([rotate_list_to_most_minimum_value(l[::-1]) for l in l_pos_edge_list_2])

        l_check_1 = [(l1==l2) or (l1==l2r) for l1, l2, l2r in zip(l_pos_list_1_sorted, l_pos_list_2_sorted, l_pos_list_2_rot_sorted)]
        assert all(l_check_1)
        assert all([(l1==l2) or (l1==l2r) for l1, l2, l2r in zip(l_pos_edge_list_1_sorted, l_pos_edge_list_2_sorted, l_pos_edge_list_2_rot_sorted)])


if __name__ == "__main__":
    # TODO 2020.03.14: add an object or a function for
    # the adding and substracting of n-dim billard table!
    
    # do_testcases_for_2d_and_n_dim_functions()

    min_dim = 2
    max_dim = 4

    # d_permutations = {i: np.array(list(itertools.permutations(range(0, i)))) for i in range(min_dim, max_dim+1)}

    PATH_FOLDER_OBJECTS = PATH_ROOT_DIR+'objs/bouncing_billard/'
    if not os.path.exists(PATH_FOLDER_OBJECTS):
        os.makedirs(PATH_FOLDER_OBJECTS)
    FILE_PATH_OBJECT = PATH_FOLDER_OBJECTS+'dim_amount_cycles.pkl.gz'

    if not os.path.exists(FILE_PATH_OBJECT):
        d_dim_amount_cycles = {}
        s_done_t_size = set()
    else:
        with gzip.open(FILE_PATH_OBJECT, 'rb') as f:
            d_objs = dill.load(f)
            d_dim_amount_cycles = d_objs['d_dim_amount_cycles']
            s_done_t_size = d_objs['s_done_t_size']
    # d_dim_amount_cycles = defaultdict(lambda: defaultdict(lambda: []))
    for i_iter in range(0, 300):
        l_size = np.random.randint(1, 20, (np.random.randint(min_dim, max_dim+1), )).tolist()
        t_size = tuple(sorted(l_size))
        dim = len(t_size)
        print("i_iter: {}, t_size: {}".format(i_iter, t_size))
        if t_size in s_done_t_size:
            print("- Ignoring t_size '{}'!".format(t_size))
            continue
        # arr_size = np.array(t_size)
        # l_size_perm = [tuple(l) for l in arr_size[d_permutations[arr_size.shape[0]]].tolist()]
        # s_size_perm = set(l_size_perm)
        # for t_size in s_size_perm:

        l_pos_list, l_pos_edge_list = calc_billard_all_possible_ball_moves_n_dim(t_size)

        amount_corner_paths = 2**(len(t_size)-1)
        amount_cycles = len(l_pos_list)-amount_corner_paths
        print("amount_corner_paths: {}".format(amount_corner_paths))
        print("amount_cycles: {}".format(amount_cycles))

        if not dim in d_dim_amount_cycles:
            d_dim_amount_cycles[dim] = {}
        d_amount_cycles = d_dim_amount_cycles[dim]
        if not amount_cycles in d_amount_cycles:
            d_amount_cycles[amount_cycles] = []  
        d_amount_cycles[amount_cycles].append(tuple(t_size))
        s_done_t_size.add(t_size)

    l_info_stats = sorted([[k1]+[sorted([(k2, len(v2)) for k2, v2 in v1.items()])] for k1, v1 in d_dim_amount_cycles.items()])
    print("l_info_stats: {}".format(l_info_stats))
    # print("d_dim_amount_cycles: {}".format(d_dim_amount_cycles))

    with gzip.open(FILE_PATH_OBJECT, 'wb') as f:
        d_objs = {}
        d_objs['d_dim_amount_cycles'] = d_dim_amount_cycles
        d_objs['s_done_t_size'] = s_done_t_size
        dill.dump(d_objs, f)

    sys.exit(0)

    d_amount_paths_for_sizes = {}
    max_size1 = 100
    max_size2 = 120
    arr = np.zeros((max_size1-2, max_size2-3), dtype=np.int)
    for size1 in range(2, max_size1):
        for size2 in range(size1+1, max_size2):
            if gcd(size1, size2)!=1:
                continue

            l_pos_list, l_pos_edge_list = calc_billard_all_possible_ball_moves_2d(size1, size2)
            amount_paths = len(l_pos_edge_list)
            print("size1: {}, size2: {}, amount_paths: {}".format(size1, size2, amount_paths))
            d_amount_paths_for_sizes[(size1, size2)] = amount_paths
            arr[size1-2, size2-3] = amount_paths

    pix_idx = arr-1
    pix_idx[arr==0] = 0
    max_idx = np.max(pix_idx)+1
    arr_colors = np.array([(0, 0, 0)]+sorted(np.random.randint(0, 256, (max_idx, 3), dtype=np.uint8).tolist()), dtype=np.uint8)
    img = Image.fromarray(arr_colors[pix_idx])
    img = img.resize((img.width*3, img.height*3))
    img.show()

    sys.exit(0)

    # w = 5
    # h = 3
    # print("w: {}, h: {}".format(w, h))
    # amount_bounces, bounces_lst = get_bounces_of_a_billard_table(w, h)
    # print("amount_bounces: {}".format(amount_bounces))
    # print("bounces_lst: {}".format(bounces_lst))

    d = {}
    d2 = defaultdict(lambda: [])
    for h in range(1, 46):
        for w in range(h+1, 46):
            v = get_bounces_amount_of_a_billard_table(h, w)
            d[(w, h)] = v
            d2[w].append(v)
            # d[(h, w)] = get_bounces_of_a_billard_table(w, h)[0]
    print("d: {}".format(d))
    lst = list(d.values())
    unique, counts = np.unique(np.array(lst), return_counts=True)
    print("unique: {}".format(unique))
    print("counts: {}".format(counts))

    l = sorted(list(d.keys()))

    l2 = [d[k] for k in l]
    # l2 = np.array([d[k] for k in l])
    print("l2: {}".format(l2))
    print("d2:")
    for k in sorted(d2.keys()):
        print("{}: {}".format(k, d2[k]))
