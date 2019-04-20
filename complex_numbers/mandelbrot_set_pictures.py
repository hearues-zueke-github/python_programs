#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys
import time

import numpy as np

from PIL import Image

import decimal
decimal.getcontext().prec = 100

from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

from ComplexDec import ComplexDec

if __name__ == "__main__":
    # c1 = ComplexDec(2, 3)
    # c2 = ComplexDec(8, -2)
    # max_iterations = 100
    
    div_factor = 1.
    
    def get_colors(max_iterations):
        dir_objs = ROOT_PATH+"objs/"
        if not os.path.exists(dir_objs):
            os.makedirs(dir_objs)

        
        max_colors = int(max_iterations/div_factor)+1
        file_colors = dir_objs+"colors.pkl.gz"
        if not os.path.exists(file_colors):
            # colors = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)
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

        # c = colors[1:][::-1].copy()
        # colors[1:] = c

        print("max_colors: {}".format(max_colors))
        print("colors.shape: {}".format(colors.shape))

        return colors
    # colors = np.array([(lambda x: [x, x, x])(int(i*((0xFF-0x10)/(max_iterations-1))+0x10)) for i in range(0, max_iterations)], dtype=np.uint8)

    limit = 2

    w_resolution = 120
    h_resolution = 100

    def get_mandelbrot_set_picture(width_half, height_half, x0, y0, max_iterations):
        print("max_iterations: {}".format(max_iterations))

        print("width_half*2: {}".format(width_half*2))
        print("height_half*2: {}".format(height_half*2))

        x1 = x0-width_half
        x2 = x0+width_half
        y1 = y0-height_half
        y2 = y0+height_half

        diff_x = x2-x1
        diff_y = y2-y1

        dx = diff_x/w_resolution
        dy = diff_y/h_resolution

        print("x1: {}, y1: {}".format(x1, y1))
        print("x2: {}, y2: {}".format(x2, y2))
        print("dx: {}".format(dx))
        print("dy: {}".format(dy))

        arr = np.empty((h_resolution, w_resolution), dtype=np.int)
        pix = np.empty((h_resolution, w_resolution, 3), dtype=np.uint8)

        for j in range(0, h_resolution):
            if j % 20 == 0:
                print("j: {}".format(j))
            y = y2-dy*(j+1)
            for i in range(0, w_resolution):
                x = x1+dx*i
                c = ComplexDec(x, y)
                z = ComplexDec(0, 0)
                # c = complex(x, y)
                # z = complex(0, 0)
                for it in range(0, max_iterations):
                    z = z*z+c
                    # if z.abs() > 2:
                    if z.abs() > limit:
                        break
                arr[j, i] = it

        min_val = np.min(arr)
        max_val = np.max(arr)
        print("min_val: {}".format(min_val))
        print("max_val: {}".format(max_val))

        # print("arr:\n{}".format(arr))

        arr_reverse = np.abs(arr-np.max(arr))
        arr_reverse = (arr_reverse/div_factor).astype(np.int)
        
        colors = get_colors(max_iterations)[:np.max(arr_reverse)+1]
        c = colors[1:][::-1].copy()
        colors[1:] = c
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
        # (Dec(2), Dec(2), Dec(-0.4), Dec(0), 100),
        # (Dec(1.5), Dec(1.5), Dec(-0.3), Dec(0.5), 150),
        # (Dec(1.0), Dec(1.0), Dec(-0.2), Dec(0.7), 200),
        # (Dec(0.5), Dec(0.5), Dec(-0.0), Dec(0.8), 250),
        # (Dec(0.2), Dec(0.2), Dec(0.1), Dec(0.85), 300),
        # (Dec('0.1'), Dec('0.1'), Dec('0.03'), Dec('0.88'), 350),
        # (Dec('0.05'), Dec('0.05'), Dec('0.03'), Dec('0.88'), 400),
        # (Dec('0.025'), Dec('0.025'), Dec('0.0155'), Dec('0.85'), 450),
        # (Dec('0.0125'), Dec('0.0125'), Dec('0.0155'), Dec('0.85'), 500),
        # (Dec('0.006125'), Dec('0.006125'), Dec('0.0154'), Dec('0.849'), 550),
        # (Dec('0.006125')/2, Dec('0.006125')/2, Dec('0.0154'), Dec('0.849'), 600),
        # (Dec('0.006125')/4, Dec('0.006125')/4, Dec('0.0154'), Dec('0.849'), 650),
        # (Dec('0.006125')/8, Dec('0.006125')/8, Dec('0.0154'), Dec('0.849'), 700),

        (Dec('0.00625'), Dec('0.00625'), Dec('0.0154'), Dec('0.849'), 100),
        (Dec('0.00625'), Dec('0.00625'), Dec('0.0154'), Dec('0.849'), 100),
    ]

    hex_letters = np.array(list('0123456789ABCDEF'))
    hash_length = 4
    rand_hash = "".join(np.random.choice(hex_letters, hash_length))
    interpolations_amount = 2
    alpha_perc = Dec(1) / interpolations_amount
    for it, (prm1, prm2) in enumerate(zip(params[:-1], params[1:]), 0):
        print("it: {}".format(it))

        width_half1, height_half1, x01, y01, max_iters1 = prm1
        width_half2, height_half2, x02, y02, max_iters2 = prm2
        
        qw = (width_half2/width_half1)**(1/Dec(interpolations_amount))
        qh = (height_half2/height_half1)**(1/Dec(interpolations_amount))
        dx = (x02-x01)/Dec(interpolations_amount)
        dy = (y02-y01)/Dec(interpolations_amount)
        dmax_iters = (max_iters2-max_iters1)/Dec(interpolations_amount)
        for it_inter in range(1, interpolations_amount+1):
            width_half = width_half1*qw**it_inter
            height_half = height_half1*qh**it_inter
            x0 = x01+dx*it_inter
            y0 = y01+dy*it_inter
            max_iters = int(max_iters1+dmax_iters*it_inter)

            start_time = time.time()
            pix = get_mandelbrot_set_picture(width_half, height_half, x0, y0, max_iters)
            end_time = time.time()

            print("end_time-start_time: {}".format(end_time-start_time))
            sys.exit(-1)

            img = Image.fromarray(pix)
            # img.show()
            img.save(dir_imgs+"mandelbrot_img_{}_lim_{:06.03f}_it_{:03}_inter{:02}.png".format(rand_hash, limit, it, it_inter))
