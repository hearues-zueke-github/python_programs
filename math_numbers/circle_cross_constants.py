#! /usr/bin/python3

# -*- coding: utf-8 -*-

# Some other needed imports

import datetime
import dill
import gzip
import os
import pdb
import re
import sys
import traceback

# Needed for excel tabels
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles.borders import Border, Side, BORDER_THIN, BORDER_MEDIUM
from openpyxl.styles import Alignment, borders, Font

import matplotlib.pyplot as plt

import decimal
from decimal import Decimal as Dec
precision = 50
decimal.getcontext().prec = precision

from PIL import Image

# Needed for DataFrame (databases etc.)
import numpy as np
import pandas as pd
import pandas.io.sql as sql

from functools import reduce

from dotmap import DotMap

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

def calc_abcd(r, dx, dy):
    a = np.sqrt(r**2-dx**2)-dy
    b = np.sqrt(r**2-dy**2)-dx
    c = np.sqrt(r**2-dy**2)+dx
    d = np.sqrt(r**2-dx**2)+dy

    return a, b, c, d


def calc_k(r, dx, dy):
    # a = np.sqrt(r**2-dx**2)-dy
    # b = np.sqrt(r**2-dy**2)-dx
    # c = np.sqrt(r**2-dy**2)+dx
    # d = np.sqrt(r**2-dx**2)+dy
    
    a = (r**2-dx**2)**(1/Dec(2))-dy
    b = (r**2-dy**2)**(1/Dec(2))-dx
    c = (r**2-dy**2)**(1/Dec(2))+dx
    d = (r**2-dx**2)**(1/Dec(2))+dy

    k = a/b-b/c

    return k


if __name__ == "__main__":
    print("Hello World!")

    r = 1

    # # create a sqr with (x, y) values
    # width = 2*r
    # height = 2*r
    # w_pix = 100
    # h_pix = 100

    # x_center = 0.
    # y_center = 0.

    # x_start = x_center-width/2
    # x_end = x_center+width/2
    # y_start = y_center-height/2
    # y_end = y_center+height/2

    # delta_x = width/w_pix
    # delta_y = height/h_pix

    # background_color = (0x00, 0x00, 0x00)
    # circle_color = (0xFF, 0xFF, 0x00)
    # pix = np.zeros((h_pix, w_pix, 3), dtype=np.uint8)

    # x = w_pix//2
    # y = h_pix-1
    # pix[y, x] = circle_color
    # y_half = h_pix//2
    # # draw a circle on the bottom right image!
    # while y>=y_half:
    #     break
    #     # x += 1

    # img = Image.fromarray(pix)
    # img.show()

    # sys.exit()

    # dy = 0.35

    # l = []
    # dxs = []
    # max_dx = np.sqrt(r**2-dy**2)
    # for dx in np.arange(0, max_dx+max_dx/5000., max_dx/5000.):
    #     k = calc_k(r, dx, dy)
    #     # l.append(k)
    #     l.append(np.abs(k))
    #     dxs.append(dx)

    # plt.figure()

    # plt.plot(dxs, l, 'b.', markersize=1.)

    # plt.show()


    r = 1

    xs_circle = []
    ys_circle = []
    for t in np.arange(0, np.pi/2, np.pi/2/100):
        xs_circle.append(r*np.cos(t))
        ys_circle.append(r*np.sin(t))

    epsilon = Dec(1)/10**(precision//2)

    xs = []
    ys = []
    n_y = 20000
    for i_y in range(0, n_y):
    # for dy in np.arange(0, 1, 0.0001):
        # find the best x for the x!
        dy = (Dec(r)*i_y)/n_y
        print("dy: {}".format(dy))
        max_dx = np.sqrt(Dec(r)**2-Dec(dy)**2)
        dx = max_dx/Dec(2)
        dx_prev = dx
        dx_inc = max_dx/Dec(4)
        i_x = 0
        while i_x < 100:
            k = calc_k(r, dx, dy)
            if k > 0:
                dx -= dx_inc
            else:
                dx += dx_inc
            if (dx_prev-dx).copy_abs() < epsilon:
                break
            dx_inc /= Dec(2)
            dx_prev = dx
            i_x += 1
        print("i_x: {}".format(i_x))
        # l = []
        # dxs = []
        # n = 500
        # for dx in np.arange(0, n):
        # # for dx in np.arange(0, max_dx, max_dx/500.):
        # # for dx in np.arange(0, max_dx, max_dx/1000.):
        #     dx = max_dx*dx/n
        #     k = calc_k(r, dx, dy)
        #     l.append(np.abs(k))
        #     dxs.append(dx)
        ys.append(dy)
        xs.append(dx)
        # xs.append(dxs[np.argmin(l)])

    xs.append(0)
    ys.append(r)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    length_circumference = np.sum(np.sqrt((xs_arr[1:]-xs_arr[:-1])**2+(ys_arr[1:]-ys_arr[:-1])**2))
    print("length_circumference: {}".format(length_circumference))

    # 1.1861938471376499167039281401611603131812648241549 100
    # 1.18651545423700853377463374765 300
    # 1.18653192429063420090976006258 600
    # 1.18654161482431847374731824616 1000
    # 1.18654655379842618007154673566 2000
    # 1.1865483618874121492795203821094595471561009200895 5000
    # 1.1865488386185807381852734463628719415420250877037 20000

    # find the least sum of the lengths!
    l = []
    for dx, dy in zip(xs, ys):
        # ly = np.sqrt(r**2-dx**2)
        # lx = np.sqrt(r**2-dy**2)
        # l.append(ly/(lx+0.0001)+lx/(ly+0.0001))
        l.append(dx)
    idx = np.argmax(l)
    # idx = np.argmin(l)
    best_x, best_y = xs[idx], ys[idx]

    plt.figure()

    plt.plot(xs, ys, 'b.', markersize=0.5)
    plt.plot(xs_circle, ys_circle, 'r.', markersize=0.5)
    plt.plot([best_x], [best_y], 'g.', markersize=15.)

    plt.axes().set_aspect('equal', 'datalim')

    plt.show()
