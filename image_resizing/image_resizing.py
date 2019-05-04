#! /usr/bin/python3

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

# import decimal
# prec = 50
# decimal.getcontext().prec = prec

# from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# import utils

if __name__ == "__main__":
    w1 = 400
    h1 = 300
    pix_orig = np.random.randint(0, 256, (h1, w1, 3), dtype=np.uint8)
    img = Image.fromarray(pix_orig)
    # img.show()

    # choose a frame
    x1 = 30
    x2 = 130
    y1 = 50
    y2 = 110

    wf = x2-x1
    hf = y2-y1

    w2 = wf+3
    h2 = hf+2

    # c = (0, 0, 0)s
    pix = pix_orig.copy()
    # pix[y1:y2, x1] = c
    # pix[y1:y2, x2-1] = c
    # pix[y1, x1:x2] = c
    # pix[y2-1, x1:x2] = c

    print("pix[:2, :3]:\n{}".format(pix[y1:y1+2, x1:x1+3]))

    img = Image.fromarray(pix)
    # img2.show()

    pix_f = pix[y1:y2, x1:x2]
    img_f = Image.fromarray(pix_f)
    # img_f.show()

    # img4 = img_f.resize((img_f.width*3, img_f.height*3))
    # img4.show()

    img_f_resize = img_f.resize((img_f.width*5, img_f.height*5))
    img_f_resize.show()
    img_f_resize.save("img_f_resize.png")

    img_2 = img_f.resize((w2, h2))
    img_2_resize = img_2.resize((img_2.width*5, img_2.height*5))
    img_2_resize.show()
    img_2_resize.save("img_2_resize.png")

    img_2_nearest = img_f.resize((w2, h2), Image.NEAREST)
    img_2_nearest_resize = img_2_nearest.resize((img_2_nearest.width*5, img_2_nearest.height*5))
    img_2_nearest_resize.save("img_2_nearest_resize.png")

    img_2_bilinear = img_f.resize((w2, h2), Image.BILINEAR)
    img_2_bilinear_resize = img_2_bilinear.resize((img_2_bilinear.width*5, img_2_bilinear.height*5))
    img_2_bilinear_resize.save("img_2_bilinear_resize.png")

    img_2_bicubic = img_f.resize((w2, h2), Image.BICUBIC)
    img_2_bicubic_resize = img_2_bicubic.resize((img_2_bicubic.width*5, img_2_bicubic.height*5))
    img_2_bicubic_resize.save("img_2_bicubic_resize.png")

    img_2_lanczos = img_f.resize((w2, h2), Image.LANCZOS)
    img_2_lanczos_resize = img_2_lanczos.resize((img_2_lanczos.width*5, img_2_lanczos.height*5))
    img_2_lanczos_resize.save("img_2_lanczos_resize.png")

    img_3 = img_2.resize((img_f.width, img_f.height))
    img_3_resize = img_3.resize((img_3.width*5, img_3.height*5))
    img_3_resize.save("img_3_resize.png")

    def get_new_color(pix, px1, py1, px2, py2):
        # print("py1: {}, py2: {}".format(py1, py2))
        # print("px1: {}, px2: {}".format(px1, px2))

        ix1 = int(px1)
        iy1 = int(py1)
        ix2 = int(px2)
        iy2 = int(py2)

        # print("iy1: {}, iy2: {}".format(iy1, iy2))
        # print("ix1: {}, ix2: {}".format(ix1, ix2))

        dpy = py2-py1
        dpx = px2-px1

        # print("dpy: {}, dpx: {}".format(dpy, dpx))
        A = dpy*dpx
        # print("A: {}".format(A))

        p0x1 = int(px1-px1%1+1)
        p0y1 = int(py1-py1%1+1)
        p0x2 = int(px2-px2%1)
        p0y2 = int(py2-py2%1)

        # print("p0y1: {}, p0y2: {}".format(p0y1, p0y2))
        # print("p0x1: {}, p0x2: {}".format(p0x1, p0x2))

        cs = []
        As = []

        # center parts
        if p0x1 != p0x2 and p0y1 != p0y2:
            # print("CENTER!")
            # sys.exit(-1)
            cs.extend(pix[p0y1:p0y2, p0x1:p0x2].reshape((-1, 3)).tolist())
            As.extend([1 for _ in range(0, p0y2-p0y1) for _ in range(0, p0x2-p0x1)])
            # pass
            # add other colors for this...

        # border parts
        if px1 < p0x1 and p0y1 < p0y2:
            # do the left loop part!
            diff = p0x1-px1
            x = p0x1-1
            for y in range(p0y1, p0y2):
                cs.append(pix[y, x])
                As.append(diff)
            # print("do the left loop part!")
            # sys.exit(-1)
            # pass
        if p0x2 < px2 and p0y1 < p0y2:
            # do the right loop part!
            diff = px2-p0x2
            x = p0x2-0
            for y in range(p0y1, p0y2):
                cs.append(pix[y, x])
                As.append(diff)
            # print("do the right loop part!")
            # sys.exit(-1)
            # pass
        if py1 < p0y1 and p0x1 < p0x2:
            # do the top loop part!
            diff = p0y1-py1
            y = p0y1-1
            for x in range(p0x1, p0x2):
                cs.append(pix[y, x])
                As.append(diff)
            # print("do the top loop part!")
            # sys.exit(-1)
            # pass
        if p0y2 < py2 and p0x1 < p0x2:
            # do the bottom loop part!
            diff = py2-p0y2
            y = p0y2-0
            for x in range(p0x1, p0x2):
                cs.append(pix[y, x])
                As.append(diff)
            # print("do the bottom loop part!")
            # sys.exit(-1)
            # pass

        # corner parts
        if px1 <= p0x1 and py1 <= p0y1:
            # do the left top part!
            # print("do the left top part!")
            c = pix[p0y1-1, p0x1-1]
            cs.append(c)
            As.append((p0y1-py1)*(p0x1-px1))
        if p0x2 < px2 and py1 < p0y1:
            c = pix[p0y1-1, p0x2-0]
            cs.append(c)
            As.append((p0y1-py1)*(px2-p0x2))
            # do the right top part!
            # print("do the right top part!")
            # sys.exit(-1)
            # pass
        if px1 < p0x1 and p0y2 < py2:
            c = pix[p0y2-0, p0x1-1]
            cs.append(c)
            As.append((py2-p0y2)*(p0x1-px1))
            # do the left bottom part!
            # print("do the left bottom part!")
            # sys.exit(-1)
            # pass
        if p0x2 < px2 and p0y2 < py2:
            c = pix[p0y2-0, p0x2-0]
            cs.append(c)
            As.append((py2-p0y2)*(px2-p0x2))
            # do the right bottom part!
            # print("do the right bottom part!")
            # sys.exit(-1)
            # pass

        # print("cs: {}".format(cs))
        # print("As: {}".format(As))

        cs = np.array(cs, dtype=np.float)
        As = np.array(As, dtype=np.float).reshape((-1, 1))

        # print("cs:\n{}".format(cs))
        # print("As:\n{}".format(As))

        c = (np.sum(cs*As, axis=0)/A).astype(np.uint8)
        # print("c: {}".format(c))

        return c
        # sys.exit(-1)

    w2 += 2
    h2 += 2

    x1 -= 1
    x2 += 1
    y1 -= 1
    y2 += 1

    wf += 2
    hf += 2


    xs = np.arange(0, w2+1)*(wf/w2)+x1
    ys = np.arange(0, h2+1)*(hf/h2)+y1

    pix_2 = np.zeros((h2, w2, 3), dtype=np.uint8)
    for j, (py1, py2) in enumerate(zip(ys[:-1], ys[1:]), 0):
        print("j: {}".format(j))
        for i, (px1, px2) in enumerate(zip(xs[:-1], xs[1:]), 0):
            c = get_new_color(pix, px1, py1, px2, py2)
            pix_2[j, i] = c


    pix_2 = pix_2[1:-1, 1:-1]

    img_2_own = Image.fromarray(pix_2)
    img_2_own_resize = img_2_own.resize((img_2_own.width*5, img_2_own.height*5))
    img_2_own_resize.show()
    img_2_own_resize.save("img_2_own_resize.png")
