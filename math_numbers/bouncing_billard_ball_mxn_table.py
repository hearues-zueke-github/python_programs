#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy
from dotmap import DotMap

from sortedcontainers import SortedSet

from collections import defaultdict

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time
from functools import reduce

from PIL import Image

import numpy as np

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
    print("h: {}, w: {}".format(h, w))
    print("lst: {}".format(lst))
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


if __name__ == "__main__":
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
