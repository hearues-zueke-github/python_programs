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


if __name__ == "__main__":
    def f(b):
        d = {1:1}
        l = [0, 1]

        # def b(n):


        def a(n):
            return a.b(a, n)
            # return (l[(n-1)]+l[(l[n-1]+l[n-2]+1)%n])%n
            # b = (a.l[n-1]+1+a.l[(a.l[n-1]+1)%n])%n
            # return (a.l[b]+a.l[(b-1)%n]+b)%n

            # return a.l[a.l[a.l[n-1]]]
        a.d = d
        a.l = l
        # a.d = {1: 1}
        # a.l = [0, 1]
        a.b = b

        # d = a.d
        # l = a.l
        return a

    def b1(a, n):
        return (a.l[(n-1)]+a.l[(a.l[n-1]+a.l[n-2]+1)%n])%n
    def b2(f, n):
        return (f.l[(n-1)]+f.l[(f.l[n-1]+f.l[n-2]+1)%n])

    a = f(b1)
    # a.b = b
    for i in range(2, 101):
        # if i%2==0:
        #     a.b = b2
        # else:
        #     a.b = b1
        a.l.append(a(i))
    print("a.l: {}".format(a.l))
    # print("l: {}".format(l))
