#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import re
import sys
import time

import itertools
import multiprocessing

import subprocess

cpu_amount = multiprocessing.cpu_count()

import numpy as np

from PIL import Image

import decimal
prec = 50
decimal.getcontext().prec = prec

from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

import utils

from ComplexDec import ComplexDec

def create_input_file(h=100, w=120, amount_interpolations=10, folder_name=None):
    s = ""

    # TODO: create from start to finish point the interpolations!
    # TODO: define m amount of interpolations in between!

    # h = 100
    # w = 120

    x1 = Dec('-0.5')
    y1 = Dec('0')

    # x2 = Dec('0.0154')
    # y2 = Dec('0.849')

    x2 = Dec('0.138331771159')
    y2 = Dec('0.6432041937823')

    w11 = Dec('3.')
    w12 = Dec('0.0000005')

    # dx = (x2-x1)/amount_interpolations
    # dy = (y2-y1)/amount_interpolations

    qw = (w12/w11)**(1/Dec(amount_interpolations))
    
    q = qw

    lx = x2-x1
    ly = y2-y1
    
    # print("q: {}".format(q))
    q_sum = np.sum([q**i for i in range(0, amount_interpolations)])
    # print("q_sum: {}".format(q_sum))
    # sys.exit(-1)
    ax = lx/q_sum
    ay = ly/q_sum
    
    # qx = (x2/x1)**(1/Dec(amount_interpolations))
    # qy = (y2/y1)**(1/Dec(amount_interpolations))

    x0 = x1-ax*q**-1
    y0 = y1-ay*q**-1

    max_iters = 500

    dir_path = "data/mandelbrotset/{}/".format(folder_name if not folder_name is None else utils.get_date_time_str())
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # output_path_template = "data/mandelbrotset/iters_nr_{number:03}.hex"
    output_path_template = dir_path+"iters_nr_{number:03}.hex"

    # "100,160,0.849,0.0154,0.0125,100,data/mandelbrotset/parameterOutput.hex"
    str_template = "{{h}},{{w}},{{y0:.0{prec}f}},{{x0:.0{prec}f}},{{w1:.0{prec}f}},{{max_iters}},{{output_path}}\n".format(prec=prec)
    for i in range(0, amount_interpolations+1):
        print("i: {}".format(i))
        y0 += ay*q**(i-1)
        x0 += ax*q**(i-1)
        # y0 = y1+dy*i
        # x0 = x1+dx*i
        w1 = w11*qw**i
        output_path = output_path_template.format(number=i)

        s += str_template.format(h=h, w=w, y0=y0, x0=x0, w1=w1, max_iters=max_iters, output_path=output_path)

    print("s:\n{}".format(s))

    lines = s.split("\n")
    lines = list(filter(lambda x: "" != x, lines))

    lines = np.random.permutation(lines).tolist()

    len_part = (len(lines)//cpu_amount)+1
    lines_parts = [lines[len_part*i:len_part*(i+1)] for i in range(0, cpu_amount)]

    # # for creating only 1 entry! (for 1 process only!)
    # lines_parts = [itertools.chain.from_iterable(lines_parts)]

    files_path = []
    for i, lines_part in enumerate(lines_parts, 1):
        file_path = "data/mandelbrotset/parameterInput_nr_{:02}.txt".format(i)
        with open(file_path, "w") as file:
            file.write("\n".join(lines_part)+"\n")
        files_path.append(file_path)

    return files_path, output_path_template


def get_colors_part():
    start_val = 0x40
    end_val = 0xFF
    def get_new_arr_part():
        return np.empty((end_val-start_val, 3), dtype=np.uint8)
    arr_vals = np.arange(start_val+1, end_val+1).astype(np.uint8)
    arr_vals_reverse = np.arange(end_val-1, start_val-1, -1).astype(np.uint8)

    vals = np.array([arr_vals_reverse, arr_vals]).astype(np.uint8)

    changing_bits = np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,1,1],[1,0,1],[1,0,0],[0,0,0]])
    # changing_bits_2 = np.vstack((changing_bits, changing_bits[:1]))
    where_idx = np.where(changing_bits[:-1]!=changing_bits[1:])[1]
    changing_bits = changing_bits[1:]
    print("vals:\n{}".format(vals))
    print("changing_bits:\n{}".format(changing_bits))
    print("where_idx:\n{}".format(where_idx))

    colors_part = np.empty((0, 3), dtype=np.uint8)
    for row, pos in zip(changing_bits, where_idx):
        arr = get_new_arr_part()
        for idx, v in enumerate(row, 0):
            if pos == idx:
                arr[:, idx] = vals[v]
            else:
                arr[:, idx] = end_val if v else start_val
        colors_part = np.vstack((colors_part, arr))
    colors_part = colors_part[::-1]
    print("colors_part:\n{}".format(colors_part))
    print("colors_part.shape: {}".format(colors_part.shape))

    return colors_part


def get_colors_part_2(last_color, n):
    min_val = 0x40
    max_val = 0xFF

    spacing = 0x8

    last_color = list(last_color)
    colors_part = []
    idxs = np.random.randint(0, 3, (n, ))

    for idx in idxs:
        v = last_color[idx]
        j = int(np.random.randint(0, 2))

        if v == min_val:
            v += int(np.random.randint(spacing//2, spacing+1))
            if v < min_val:
                v = min_val
            last_color[idx] = v
        elif v == max_val:
            v -= int(np.random.randint(spacing//2, spacing+1))
            if v < min_val:
                v = min_val
            last_color[idx] = v
        elif j == 0:    
            v -= int(np.random.randint(spacing//2, spacing+1))
            if v < min_val:
                v = min_val
            last_color[idx] = v
        else:
            v += int(np.random.randint(spacing//2, spacing+1))
            if v < min_val:
                v = min_val
            last_color[idx] = v

        colors_part.append(list(last_color))

    colors_part = np.array(colors_part).astype(np.uint8)
    # print("colors_part:\n{}".format(colors_part))
    # sys.exit(-1)

    return colors_part


div_factor = 1.
def get_colors(max_iterations):
    dir_objs = ROOT_PATH+"objs/"
    if not os.path.exists(dir_objs):
        os.makedirs(dir_objs)

    max_colors = max_iterations+1
    # max_colors = int(max_iterations/div_factor)+1

    file_colors = dir_objs+"colors.pkl.gz"
    if not os.path.exists(file_colors):
        # colors = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)

        # TODO: add gradient colors to the colors array!
        # colors = np.random.randint(0, 256, (max_colors, 3), dtype=np.uint8)
        # colors[0] = (0x00, 0x00, 0x00)
        colors = np.zeros((1, 3), dtype=np.uint8)
        # colors_part = get_colors_part_2(np.random.randint(0x40, 0xFF, (3, )).tolist(), 100)
        colors_part = get_colors_part_2([0, 0, 0], 100)
        # colors_part = get_colors_part()
        colors = np.vstack((colors, colors_part))

        with gzip.open(file_colors, "wb") as file:
            dill.dump(colors, file)
    else:
        with gzip.open(file_colors, "rb") as file:
            colors = dill.load(file)

        if colors.shape[0] < max_colors:
            print("Need to extend colors!")
            colors_part = get_colors_part_2(colors[-1].tolist(), 100)
            # colors_part = get_colors_part()
            colors = np.vstack((colors, colors_part))
            # colors_next = np.random.randint(0, 256, (max_colors-colors.shape[0], 3), dtype=np.uint8)
            # colors = np.vstack((colors, colors_next))

            with gzip.open(file_colors, "wb") as file:
                dill.dump(colors, file)

    print("max_colors: {}".format(max_colors))
    # print("colors.shape: {}".format(colors.shape))

    return colors


def get_mandelbrot_set_picture(h_resolution, w_resolution, arr, max_iterations):
    pix = np.empty((h_resolution, w_resolution, 3), dtype=np.uint8)

    min_val = np.min(arr)
    max_val = np.max(arr)
    print("min_val: {}".format(min_val))
    print("max_val: {}".format(max_val))

    arr_reverse = np.abs(arr-np.max(arr))
    # arr_reverse = (arr_reverse/div_factor).astype(np.int)
    colors = get_colors(max_iterations)
    colors = np.vstack((colors[0:1], colors[min_val:np.max(max_val)]))
    print("colors.shape: {}".format(colors.shape))
    print("arr.shape: {}".format(arr.shape))
    c = colors[1:][::-1].copy()
    colors[1:] = c
    print("type(arr_reverse[0, 0]): {}".format(type(arr_reverse[0, 0])))
    pix = colors[arr_reverse]

    return pix


def do_subprocesses(files_path):
    sub_procs = []

    print("Starting the processes!")
    for i, file_path in enumerate(files_path, 1):
        params = re.sub(" +", " ", file_path).strip().split(" ")
        params = ['./calc_mandelbrot_set_iterations.o']+params+['ProcNr.: {proc_num}, '.format(proc_num=i)]
        sub_proc = subprocess.Popen(params)
        sub_procs.append(sub_proc)

    print("Wait until finshed!")
    for sub_proc in sub_procs:
        sub_proc.communicate()


if __name__ == "__main__":
    # folder_name = "Y2019_m04_d21_H10_M54_S46_f391295"
    folder_name = utils.get_date_time_str()

    h = 350
    w = 450

    amount_interpolations=100
    files_path, output_path_template = create_input_file(h=h, w=w, amount_interpolations=amount_interpolations, folder_name=folder_name)
    # # sys.exit(-1)

    do_subprocesses(files_path)

    # call the cpp program for creating the arr

    print("Create now the images!")

    # load the arr and create the pix and later img png files!
    
    # input_path_template = output_path_template
    input_path_template = "data/mandelbrotset/{}/iters_nr_{{number:03}}.hex".format(folder_name)

    dir_pictures = "images/manedlbrotset/{}/".format(folder_name)
    if not os.path.exists(dir_pictures):
        os.makedirs(dir_pictures)

    for i in range(0, amount_interpolations+1):
        print("Doing i: {}".format(i))
        input_path = input_path_template.format(number=i)

        with open(input_path, "rb") as file:
            arr = np.fromfile(file, np.uint8)

        params = arr[:6].view(np.uint16)
        # print("params: {}".format(params))

        max_iters, h, w = params
        arr = arr[6:].view(np.uint16).astype(np.int)
        # print("np.min(arr): {}".format(np.min(arr)))
        # print("np.max(arr): {}".format(np.max(arr)))
        # arr = (arr>>8)|(arr<<8)
        # arr2 = arr[6:].view(np.uint16).astype('<u2')
        # print("arr[:10]: {}".format(arr[:10]))
        # print("arr2[:10]: {}".format(arr2[:10]))
        arr = arr.reshape((h, w))

        pix = get_mandelbrot_set_picture(h, w, arr, max_iters)

        img = Image.fromarray(pix)

        img.save(dir_pictures+"mandelbrot_pic_nr_{:03}.png".format(i))

    current_dir = os.getcwd()
    os.chdir(ROOT_PATH+dir_pictures)
    print("Create a gif image!")
    os.system("convert -delay 10 -loop 0 *.png animated.gif")
    os.chdir(current_dir)

    sys.exit(-1)



    with open("data/mandelbrotset/parameterOutput.hex", "rb") as file:
        arr = np.fromfile(file, np.uint8)

    params = arr[:4].view(np.uint16)
    print("params: {}".format(params))

    h, w = params
    arr = arr[4:].view(np.uint16)
    arr = arr.reshape((h, w))

    print("arr.shape: {}".format(arr.shape))

    print("arr.shape:\n{}".format(arr.shape))
    sys.exit(-1)

    # c1 = ComplexDec(2, 3)
    # c2 = ComplexDec(8, -2)
    # max_iterations = 100
    
    # colors = np.array([(lambda x: [x, x, x])(int(i*((0xFF-0x10)/(max_iterations-1))+0x10)) for i in range(0, max_iterations)], dtype=np.uint8)

    limit = 2

    w_resolution = 120
    h_resolution = 100
    def get_mandelbrot_set_picture(width_half, height_half, x0, y0, max_iterations):
        # print("max_iterations: {}".format(max_iterations))

        # print("width_half*2: {}".format(width_half*2))
        # print("height_half*2: {}".format(height_half*2))

        x1 = x0-width_half
        x2 = x0+width_half
        y1 = y0-height_half
        y2 = y0+height_half

        diff_x = x2-x1
        diff_y = y2-y1

        dx = diff_x/w_resolution
        dy = diff_y/h_resolution

        # print("x1: {}, y1: {}".format(x1, y1))
        # print("x2: {}, y2: {}".format(x2, y2))
        # print("dx: {}".format(dx))
        # print("dy: {}".format(dy))

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
        # print("min_val: {}".format(min_val))
        # print("max_val: {}".format(max_val))

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
