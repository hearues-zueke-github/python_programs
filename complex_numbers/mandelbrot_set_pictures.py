#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

import numpy as np

from PIL import Image

import decimal
decimal.getcontext().prec = 20

from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

from ComplexDec import ComplexDec

if __name__ == "__main__":
    # c1 = ComplexDec(2, 3)
    # c2 = ComplexDec(8, -2)
    max_iterations = 950
    
    dir_objs = ROOT_PATH+"objs/"
    if not os.path.exists(dir_objs):
        os.makedirs(dir_objs)

    max_colors = max_iterations//2+1
    file_colors = dir_objs+"colors.pkl.gz"
    if not os.path.exists(file_colors):
        colors = np.random.randint(0, 256, (max_colors, 3), dtype=np.uint8)
        colors[0] = (0x00, 0x00, 0x00)
        with gzip.open(file_colors, "wb") as file:
            dill.dump(colors, file)
    else:
        with gzip.open(file_colors, "rb") as file:
            colors = dill.load(file)

        if colors.shape[0] < max_colors:
            print("Need to extend colors!")
            colors_next = np.random.randint(0, 256, (max_colors-colors.shape[0], 3), dtype=np.uint8)
            colors = np.vstack((colors, colors_next))

            with gzip.open(file_colors, "wb") as file:
                dill.dump(colors, file)

    # colors = np.array([(lambda x: [x, x, x])(int(i*((0xFF-0x10)/(max_iterations-1))+0x10)) for i in range(0, max_iterations)], dtype=np.uint8)

    def get_mandelbrot_set_picture(width_half, height_half, x0, y0):
        x1 = x0-width_half
        x2 = x0+width_half
        y1 = y0-height_half
        y2 = y0+height_half

        diff_x = x2-x1
        diff_y = y2-y1


        w_resolution = 200
        h_resolution = 200

        dx = diff_x/w_resolution
        dy = diff_y/h_resolution

        print("x1: {}, y1: {}".format(x1, y1))
        print("x2: {}, y2: {}".format(x2, y2))

        arr = np.empty((h_resolution, w_resolution), dtype=np.int)
        pix = np.empty((h_resolution, w_resolution, 3), dtype=np.uint8)

        for j in range(0, h_resolution):
            if j % 100 == 0:
                print("j: {}".format(j))
            y = y2-dy*j
            for i in range(0, w_resolution):
                x = x1+dx*i
                # c = ComplexDec(x, y)
                # z = ComplexDec(0, 0)
                c = complex(x, y)
                z = complex(0, 0)
                for it in range(0, max_iterations):
                    z = z*z+c
                    # if z.abs() > 2:
                    if abs(z) > 2:
                        break
                arr[j, i] = it

        min_val = np.min(arr)
        max_val = np.max(arr)
        print("min_val: {}".format(min_val))
        print("max_val: {}".format(max_val))

        arr_reverse = np.abs(arr-np.max(arr))
        arr_reverse = (arr_reverse/2).astype(np.int)
        pix = colors[arr_reverse]

        return pix


    dir_imgs = ROOT_PATH+"images/"
    if not os.path.exists(dir_imgs):
        os.makedirs(dir_imgs)

    width_half = Dec(2)
    height_half = Dec(2)

    x0 = Dec(-0.4)
    y0 = Dec(0)

    params = [
        (Dec(2), Dec(2), Dec(-0.4), Dec(0)),
        (Dec(1.5), Dec(1.5), Dec(-0.3), Dec(0.5)),
        (Dec(1.0), Dec(1.0), Dec(-0.2), Dec(0.7)),
        (Dec(0.5), Dec(0.5), Dec(-0.0), Dec(0.8)),
        (Dec(0.2), Dec(0.2), Dec(0.1), Dec(0.85)),
    ]

    for it, (width_half, height_half, x0, y0) in enumerate(params, 0):
        print("it: {}".format(it))
        pix = get_mandelbrot_set_picture(width_half, height_half, x0, y0)

        img = Image.fromarray(pix)
        img.save(dir_imgs+"mandelbrot_img_{:03}.png".format(it))
