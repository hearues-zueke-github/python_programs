#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import colorsys
import datetime
import dill
import gzip
import os
import shutil
import string
import subprocess
import sys
import time
import traceback

import numpy as np

import matplotlib.pyplot as plt

from dotmap import DotMap

from indexed import IndexedOrderedDict
from collections import OrderedDict
from dotmap import DotMap
from PIL import Image, ImageDraw, ImageFont

all_symbols_16 = np.array(list("0123456789ABCDEF"))
def get_random_string_base_16(n):
    l = np.random.randint(0, 16, (n, ))
    return "".join(all_symbols_16[l])

def get_date_time_str():
    dt = datetime.datetime.now()
    return "Y{:04}_m{:02}_d{:02}_H{:02}_M{:02}_S{:02}_f{:06}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
    # return "{:Y%Ym%md%dH%HM%MS%Sf%f}".format(datetime.datetime.now())

def check_create_folders(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

if __name__ == "__main__":
    main_folder = "images/"
    check_create_folders(main_folder)

    max_length = 20.
    modulo = 10.
    n = 400
    xs = np.arange(0, n)/n*max_length-max_length/2
    ys = np.arange(0, n)/n*max_length-max_length/2

    # print("xs: {}".format(xs))

    funcs = [
        lambda x, y: x*y*np.cos(y)+1+np.sin(x),
    ]

    hsv_to_rgb_vectorized = np.vectorize(colorsys.hsv_to_rgb)

    def get_random_func():
        params = np.random.random((3, 9))-0.5
        cols_value = [10, 2, 2, 1, 0.2, 0.2, 0.1, 0.1, 0.05]
        # for i, v in enumerate(cols_value):
        #     params[:, i] = params[:, i]*v
        # params *= cols_value

        return lambda x, y: (
            params[0][0]+params[0][1]*np.sin(x)+params[0][2]*np.sin(y)+np.sin(x*params[0][3]+y*params[0][4]+params[0][6])*(x*params[0][7]+y*params[0][8]), # +x*y*params[0][3]+x**2*params[0][4]+y**2*params[0][5]+x**2*y*params[0][6]+x*y**2*params[0][7]+x**2*y**2*params[0][8],
            params[1][0]+params[1][1]*np.sin(x)+params[1][2]*np.sin(y)+np.sin(x*params[1][3]+y*params[1][4]+params[1][6])*(x*params[1][7]+y*params[1][8]), # +x*y*params[1][3]+x**2*params[1][4]+y**2*params[1][5]+x**2*y*params[1][6]+x*y**2*params[1][7]+x**2*y**2*params[1][8],
            params[2][0]+params[2][1]*np.sin(x)+params[2][2]*np.sin(y)+np.sin(x*params[2][3]+y*params[2][4]+params[2][6])*(x*params[2][7]+y*params[2][8]), # +x*y*params[2][3]+x**2*params[2][4]+y**2*params[2][5]+x**2*y*params[2][6]+x*y**2*params[2][7]+x**2*y**2*params[2][8],
        )

        # return lambda x, y: (
        #     params[0][0]+x*params[0][1]+y*params[0][2]+x*y*params[0][3]+x**2*params[0][4]+y**2*params[0][5]+x**2*y*params[0][6]+x*y**2*params[0][7]+x**2*y**2*params[0][8],
        #     params[1][0]+x*params[1][1]+y*params[1][2]+x*y*params[1][3]+x**2*params[1][4]+y**2*params[1][5]+x**2*y*params[1][6]+x*y**2*params[1][7]+x**2*y**2*params[1][8],
        #     params[2][0]+x*params[2][1]+y*params[2][2]+x*y*params[2][3]+x**2*params[2][4]+y**2*params[2][5]+x**2*y*params[2][6]+x*y**2*params[2][7]+x**2*y**2*params[2][8],
        # )

        # return lambda x, y: (
        #     x,
        #     y+2,
        #     x+y-3,
        # )

    rgb_func = get_random_func()
        # x+y*4+3,
        # -x+np.sin(y)-x*np.cos(y+x)
    # )

    arr_x = np.zeros((ys.shape[0], xs.shape[0]))
    arr_y = np.zeros((ys.shape[0], xs.shape[0]))

    arr_x[:] = xs
    arr_y[:] = ys.reshape((-1, 1))

    # rgb_floats = np.vectorize(funcs[0])(arr_x, arr_y)
    rgb_floats = np.dstack(np.vectorize(rgb_func)(arr_x, arr_y))
    print("rgb_floats.shape: {}".format(rgb_floats.shape))
    print("rgb_floats.dtype: {}".format(rgb_floats.dtype))

    # r_channel = rgb_floats[:, :, 0]
    # g_channel = rgb_floats[:, :, 1]
    # b_channel = rgb_floats[:, :, 2]

    # rgb_floats %= 1.
    argmax = np.argmax(np.sum(rgb_floats**2, axis=1))
    print("argmax: {}".format(argmax))

    mins = np.min(np.min(rgb_floats, axis=0), axis=0)
    maxs = np.max(np.max(rgb_floats, axis=0), axis=0)
    print("mins: {}".format(mins))
    print("maxs: {}".format(maxs))

    arr_hsv_1 = (rgb_floats-mins)/(maxs-mins)
    arr_rgb_1 = np.dstack(hsv_to_rgb_vectorized(arr_hsv_1[:, :, 0], arr_hsv_1[:, :, 1], arr_hsv_1[:, :, 2]))

    pix = (arr_rgb_1*255.9).astype(np.uint8)

    img = Image.fromarray(pix)
    img.show()

    # make a pseudo complex function plot!
    # np.

    imgs = []
    axises = [(0, 1), (0, 2), (1, 2)]

    for ax1, ax2 in axises:
        arr_rg = 1j * rgb_floats[:, :, ax2] + rgb_floats[:, :, ax1]
        arr_angle = np.angle(arr_rg)
        arr_abs = np.abs(arr_rg)
        arr_abs
        idx = arr_angle < 0
        arr_angle[idx] = arr_angle[idx]+np.pi*2
        # arr_angle /= np.max(arr_angle)
        arr_angle /= (np.pi*2)

        arr_hsv = np.dstack((arr_angle, np.ones(arr_angle.shape), np.ones(arr_angle.shape)))

        arr_rgb = (np.dstack(hsv_to_rgb_vectorized(arr_hsv[:, :, 0], arr_hsv[:, :, 1], arr_hsv[:, :, 2]))*255.9).astype(np.uint8)

        img2 = Image.fromarray(arr_rgb)
        # img2.show()
        imgs.append(img2)

    for img in imgs:
        img.show()
