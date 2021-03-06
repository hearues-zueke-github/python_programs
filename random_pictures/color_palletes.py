#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip
import mmap
import os
import re
import sys
import time

import itertools
import multiprocessing

from multiprocessing import Process, Pipe
# from multiprocessing import shared_memory # in python3.8 available!

import matplotlib
import matplotlib.pyplot as plt

import subprocess

import numpy as np

from PIL import Image

# import decimal
# prec = 50
# decimal.getcontext().prec = prec

# from decimal import Decimal as Dec

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# import utils

def random_256_256_pallete():
    pix = np.zeros((256, 256, 3), dtype=np.uint8)

    ns_x = np.arange(0, 16)*16
    ns_x[-1] = 255

    ns_x_v = np.vstack([ns_x for _ in range(0, 16)]).reshape((-1, ))
    ns_x_h = np.hstack([ns_x for _ in range(0, 16)])
    
    ns_y = np.arange(0, 256).reshape((-1, 1))
    
    pix[:, :, 1] = ns_y
    pix[:, :, 2] = ns_x_v
    pix[:, :, 0] = ns_x_h

    # pix = pix.transpose(1, 0, 2)
    pix = ( pix
        .reshape((16, 256, 16, 3))
        .transpose(1, 0, 2, 3)
        .reshape((16, 16, 16, 16, 3))
        .transpose(1, 0, 2, 3, 4)
        .reshape((256, 16, 16, 3))
        .transpose(1, 0, 2, 3)
        .reshape((256, 256, 3))
    )
    
    img = Image.fromarray(pix)
    img.show()


def random_4096_4096_pallete():
    print("Creating 256x256x256x3 color array!")
    pix = np.empty((256, 256, 256, 3), dtype=np.uint8)
    ns = np.arange(0, 256).astype(np.uint8)

    pix[:, :, :, 0] = ns
    pix[:, :, :, 1] = ns.reshape((-1, 1))
    pix[:, :, :, 2] = ns.reshape((-1, 1, 1))

    pix2 = ( pix
        .reshape((16, 16, 256, 256, 3))
        .transpose(0, 1, 3, 2, 4)
        .reshape((16, 4096, 256, 3))
        .transpose(0, 2, 1, 3)
        .reshape((4096, 4096, 3))
    )
    # img2 = Image.fromarray(pix2)
    # img2.save("all_rgb_4096_4096.png")

    print("Mixing in x direction!")
    # idxs_x = np.array([np.random.permutation(np.arange(0, 4096)) for _ in range(0, 4096)])
    # idxs_y = np.tile(np.arange(0, 4096), 4096).flatten()

    pix3 = np.array([row[np.random.permutation(np.arange(0, 4096))] for row in pix2], dtype=np.uint8)
    # pix3 = pix[(idxs_y.flatten(), idxs_x.flatten())]
    # img3 = Image.fromarray(pix3)
    # img3.save("all_rgb_4096_4096_x_permutation.png")

    print("Mixing in y direction!")

    pix4 = np.array([row[np.random.permutation(np.arange(0, 4096))] for row in pix3.transpose(1, 0, 2)], dtype=np.uint8) # .transpose(1, 0, 2)
    img4 = Image.fromarray(pix4)
    image_name_all_rgb_mixed = "all_rgb_4096_4096_x_y_permutation.png"
    print("Saving image as '{}'!".format(image_name_all_rgb_mixed))

    img4.save(image_name_all_rgb_mixed)

    return pix4


def approx_image_with_all_rgb():
    pix_try_nr = 11
    image_name = "SDvision_ramses_disksAMR8HR00019_redToWhite_4096x4096.jpg"
    # image_name_all_rgb_mixed = "all_rgb_4096_4096_x_y_permutation.png"
    image_name_all_rgb_mixed = "SDvision_ramses_disksAMR8HR00019_redToWhite_4096x4096_rgb_nearest_try_{:02}.png".format(pix_try_nr)
    # image_name_all_rgb_mixed = "all_rgb_4096_4096_x_y_permutation.png"

    img = Image.open(image_name)
    pix = np.array(img)
    # img.show()

    # if not os.path.exists(image_name_all_rgb_mixed):
    #     random_4096_4096_pallete()
    img_all_rgb = Image.open(image_name_all_rgb_mixed)
    pix_all_rgb = np.array(img_all_rgb)
    # img_all_rgb = Image.fromarray(pix_all_rgb)
    # img_all_rgb.show()

    print("convert rgb_to_hsv for pix")
    pix_hsv = matplotlib.colors.rgb_to_hsv(pix)
    print("convert rgb_to_hsv for pix_all_rgb")
    pix_all_rgb_hsv = matplotlib.colors.rgb_to_hsv(pix_all_rgb)

    # print("sort pix_hsv")
    # idx_pix = np.argsort(np.roll(pix_hsv, 2, 1).reshape((-1, )).view("f4,f4,f4"))
    # print("sort pix_all_rgb_hsv")
    # idx_pix_all_rgb = np.argsort(np.roll(pix_all_rgb_hsv, 2, 1).reshape((-1, )).view("f4,f4,f4"))

    # print("get new sorted image!")
    # pix_all_rgb_sorted = np.empty((4096*4096, 3), dtype=np.uint8)
    # pix_all_rgb_sorted[idx_pix] = pix_all_rgb.reshape((4096*4096, 3))[idx_pix_all_rgb]

    # pix_all_rgb_sorted = pix_all_rgb_sorted.reshape((4096, 4096, 3))
    # img_all_rgb_sorted = Image.fromarray(pix_all_rgb_sorted)
    # img_all_rgb_sorted.save("all_rgb_sorted_1_2_0.png")

    # sys.exit(-2)

    # globals()["pix"] = pix
    # globals()["pix_all_rgb"] = pix_all_rgb

    # globals()["pix_hsv"] = pix_hsv
    # globals()["pix_all_rgb_hsv"] = pix_all_rgb_hsv

    # sys.exit(-1)

    # pix1 = pix.reshape((-1, 3)).astype(np.double)
    # pix_all_rgb1 = pix_all_rgb.reshape((-1, 3)).astype(np.double)
    # idxs_orig = np.arange(0, pix1.shape[0])

    # shift_amount = 10
    # pix_all_rgb1_shift

    # np.sum(pix1-pix_all_rgb1, axis=1)

    # sys.exit(-1)

    m = 4
    # n = 3000000
    # xs1 = np.random.randint(0, 4096, (n, m))
    # # xs2 = np.random.randint(0, 4096, (n, m))
    # ys1 = np.random.randint(0, 4096, (n, m))
    # # ys2 = np.random.randint(0, 4096, (n, m))

    print("Create 4096x4096 idxs!")
    idxs_4096_4096 = np.tile(np.arange(0, 4096), 4096).flatten()
    def get_xs_ys():
        xs = np.random.permutation(idxs_4096_4096).copy()
        ys = np.random.permutation(idxs_4096_4096).copy()
        return xs, ys
    # xs1 = np.random.permutation(idxs_4096_4096).copy()
    # ys1 = np.random.permutation(idxs_4096_4096).copy()

    xs1, ys1 = get_xs_ys()
    for _ in range(0, 2):
        xs, ys = get_xs_ys()
        xs1 = np.vstack((xs1, xs))
        ys1 = np.vstack((ys1, ys))

    xs1 = xs1.reshape((-1, m))
    ys1 = ys1.reshape((-1, m))

    n = xs1.shape[0]

    print("xs1.shape: {}".format(xs1.shape))
    print("ys1.shape: {}".format(ys1.shape))
    print("n: {}".format(n))

    # sys.exit(-1)

    # tries_x = 0
    # idx_x = np.all(xs1!=xs2, axis=1)

    # print("idx_x: {}".format(idx_x))
    # print("idx_x.shape: {}".format(idx_x.shape))
    # print("np.sum(~idx_x): {}".format(np.sum(~idx_x)))

    # while np.sum(~idx_x) > 0:
    #     xs1_rest = xs1[idx_x]
    #     xs2_rest = xs2[idx_x]
    #     length = xs1_rest.shape[0]

    #     xs1 = np.vstack((xs1_rest, np.random.randint(0, 4096, (n-length, m))))
    #     xs2 = np.vstack((xs2_rest, np.random.randint(0, 4096, (n-length, m))))

    #     idx_x = np.any(xs1!=xs2, axis=1)
    #     tries_x += 1
    #     print("tries_x for xs1 and xs2: {}".format(tries_x))
    #     print("xs1_rest.shape: {}".format(xs1_rest.shape))

    # tries_y = 0
    # idx_y = np.all(ys1!=ys2, axis=1)
    # while np.sum(~idx_y) > 0:
    #     ys1_rest = ys1[idx_y]
    #     ys2_rest = ys2[idx_y]
    #     length = ys1_rest.shape[0]

    #     ys1 = np.vstack((ys1_rest, np.random.randint(0, 4096, (n-length, m))))
    #     ys2 = np.vstack((ys2_rest, np.random.randint(0, 4096, (n-length, m))))

    #     idx_y = np.all(ys1!=ys2, axis=1)
    #     tries_y += 1
    #     print("tries_y for ys1 and ys2: {}".format(tries_y))
    #     print("ys1_rest.shape: {}".format(ys1_rest.shape))

    pix = pix.astype(np.double)
    pix_all_rgb = pix_all_rgb.astype(np.double)

    changed_colors = 0
    stayed_colors = 0
    for it in range(0, n):
        px1 = xs1[it]
        py1 = ys1[it]
        # px2 = xs2[it]
        # py2 = ys2[it]

        cs_img_1 = pix[(py1, px1)]
        # cs_img_2 = pix[(py2, px2)]

        cs_rgb_1 = pix_all_rgb[(py1, px1)]
        # cs_rgb_2 = pix_all_rgb[(py2, px2)]

        # print("px1:\n{}".format(px1))
        # print("py1:\n{}".format(py1))

        # print("px2:\n{}".format(px2))
        # print("py2:\n{}".format(py2))


        # print("cs_img_1:\n{}".format(cs_img_1))
        # print("cs_rgb_1:\n{}".format(cs_rgb_1))
        
        # print("cs_img_2:\n{}".format(cs_img_2))
        # print("cs_rgb_2:\n{}".format(cs_rgb_2))

        diff_i1_r1 = np.abs(cs_img_1.reshape((-1, 1, 3))-cs_rgb_1)
        # print("diff_i1_r1.shape: {}".format(diff_i1_r1.shape))
        
        # diff_i2_r2 = np.abs(cs_img_2.reshape((-1, 1, 3))-cs_rgb_2)
        # diff_i1_r2 = np.abs(cs_img_1.reshape((-1, 1, 3))-cs_rgb_2)
        # diff_i2_r1 = np.abs(cs_img_2.reshape((-1, 1, 3))-cs_rgb_1)
        # print("diff_i2_r2.shape: {}".format(diff_i2_r2.shape))
        # print("diff_i1_r2.shape: {}".format(diff_i1_r2.shape))
        # print("diff_i2_r1.shape: {}".format(diff_i2_r1.shape))

        diff_sum_i1_r1 = np.sum(diff_i1_r1, axis=2)
        # print("diff_sum_i1_r1:\n{}".format(diff_sum_i1_r1))
        
        # diff_sum_i2_r2 = np.sum(diff_i2_r2, axis=2)
        # diff_sum_i1_r2 = np.sum(diff_i1_r2, axis=2)
        # diff_sum_i2_r1 = np.sum(diff_i2_r1, axis=2)

        # print("diff_sum_i1_r2:\n{}".format(diff_sum_i1_r2))
        # print("")
        # print("diff_sum_i2_r2:\n{}".format(diff_sum_i2_r2))
        # print("diff_sum_i2_r1:\n{}".format(diff_sum_i2_r1))

        min_idx = np.argmin(diff_sum_i1_r1, axis=0)
        # print("min_idx: {}".format(min_idx))

        idx_sort = np.argsort(min_idx)
        idx_orig = np.arange(0, m)

        same_idx = np.sum(idx_sort==idx_orig)
        stayed_colors += same_idx
        changed_colors += m-same_idx
        # print("idx_sort: {}".format(idx_sort))

        pix_all_rgb[(py1, px1)] = cs_rgb_1.copy()[idx_sort]

        # idxs = np.vstack((min_idx, np.arange(0, m))).T
        # print("idxs:\n{}".format(idxs))

        # diff_tbl = (diff_sum_i1_r1+diff_sum_i2_r2)-(diff_sum_i1_r2+diff_sum_i2_r1)
        # print("diff_tbl:\n{}".format(diff_tbl))

        # max_idx_axis_0 = np.argmax(diff_tbl, axis=0)
        # max_idx_axis_1 = np.argmax(diff_tbl, axis=1)

        # print("max_idx_axis_0: {}".format(max_idx_axis_0))
        # print("max_idx_axis_1: {}".format(max_idx_axis_1))

        # max_idx_1 = np.argmax(diff_sum_i1_r1-diff_sum_i1_r2, axis=0)
        # max_idx_2 = np.argmax(diff_sum_i2_r2-diff_sum_i2_r1, axis=0)

        # print("max_idx_1:\n{}".format(max_idx_1))
        # print("max_idx_2:\n{}".format(max_idx_2))


        # sys.exit(-1)

        # np.sum(np.abs(cs_img_1-cs_rgb_1))

        # # c_img_1 = pix[y1, x1]
        # # c_img_2 = pix[y2, x2]
        # # c_rgb_1 = pix_all_rgb[y1, x1].copy()
        # # c_rgb_2 = pix_all_rgb[y2, x2].copy()

        # if np.sqrt(np.sum((c_img_1-c_rgb_2)**2))+np.sqrt(np.sum((c_img_2-c_rgb_1)**2)) < np.sqrt(np.sum((c_img_1-c_rgb_1)**2)) < np.sqrt(np.sum((c_img_2-c_rgb_2)**2)):
        # # if np.sum(np.abs(c_img_1-c_rgb_2))+np.sum(np.abs(c_img_2-c_rgb_1)) < np.sum(np.abs(c_img_1-c_rgb_1))+np.sum(np.abs(c_img_2-c_rgb_2)):
        #     pix_all_rgb[y1, x1] = c_rgb_2
        #     pix_all_rgb[y2, x2] = c_rgb_1
        #     changed_colors += 1
        # else:
        #     stayed_colors += 1
        # # x1 = int(np.random.randint(0, 4096))
        # # x2 = int(np.random.randint(0, 4096))
        # # while x1 == x2:
        it += 1
        if it % 100000 == 0:
            print("it: {}".format(it))
            print("stayed_colors: {}, changed_colors: {}".format(stayed_colors, changed_colors))

    pix_all_rgb = pix_all_rgb.astype(np.uint8)
    img_all_rgb = Image.fromarray(pix_all_rgb)
    # img_all_rgb.show()
    image_name_all_rgb_mixed_new = "SDvision_ramses_disksAMR8HR00019_redToWhite_4096x4096_rgb_nearest_try_{:02}.png".format(pix_try_nr+1)
    img_all_rgb.save(image_name_all_rgb_mixed_new)


def find_best_fitting_image_for_random_pixels():
    images_amount = 4
    image_path_template = "images/img_4096_4096_nr_{num}.{extension}"
    # image_path_template = "images/img_400_300_nr_{num}.{extension}"

    img_paths = []

    if not os.path.exists("images/"):
        os.makedirs("images/")

    for i in range(1, images_amount+1):
        image_path = image_path_template.format(num=i, extension="png")
        if os.path.exists(image_path):
            img_paths.append(image_path)
            continue
        
        image_path_jpg = image_path_template.format(num=i, extension="jpg")
        if os.path.exists(image_path_jpg):
            img = Image.open(image_path_jpg)
            image_path_png = image_path_template.format(num=i, extension="png")
            img.save(image_path_png)
            continue

    print("img_paths:\n{}".format(img_paths))

    if len(img_paths) == 0:
        return

    choosen_img_path = img_paths[1]
    img_orig = Image.open(choosen_img_path)
    pix_orig = np.array(img_orig)
    img_orig.show()

    # try different approaches for random pixels, e.g. take average of each row or col and create the new pix_rgb!
    
    # take n different pixels from pix_orig and do np.tile until you get w x h image again
    # h, w, channels = pix_orig.shape
    # n = 16
    # colors_choosen = pix_orig.reshape((-1, channels))[np.random.permutation(np.arange(0, h*w))[:n]]
    # # pix_rgb = np.repeat(colors_choosen.reshape((-1, )), ((h*w)//n+1)*n).reshape((-1, 3))[:h*w*channels].reshape(pix_orig.shape)
    # pix_rgb = np.tile(colors_choosen.reshape((-1, )), (h*w)//n+1)[:h*w*channels].reshape(pix_orig.shape)

    # mean of each col and create a full new image of cols mean value pixels!
    # cols_mean = np.mean(pix_orig.astype(np.double), axis=0).astype(np.uint8)
    # print("cols_mean.shape: {}".format(cols_mean.shape))
    # pix_rgb = np.tile(cols_mean.reshape((-1, )), pix_orig.shape[0]).reshape(pix_orig.shape)

    # mean of each row and create a full new image of rows mean value pixels!
    # cols_mean = np.mean(pix_orig.astype(np.double), axis=1).astype(np.uint8)
    # print("cols_mean.shape: {}".format(cols_mean.shape))
    # pix_rgb = np.tile(cols_mean.reshape((-1, )), pix_orig.shape[1]).reshape(pix_orig.shape)

    # random pixels
    pix_rgb = np.random.randint(0, 256, pix_orig.shape).astype(np.uint8)

    print("convert from rgb to hsv for pix_orig")
    pix_orig_hsv = np.flip(matplotlib.colors.rgb_to_hsv(pix_orig).reshape((-1, 3)), 1)
    # pix_orig_hsv = np.roll(matplotlib.colors.rgb_to_hsv(pix_orig).reshape((-1, 3)), 1, axis=1)
    # pix_orig_hsv = matplotlib.colors.rgb_to_hsv(pix_orig)

    print("convert from rgb to hsv for pix_rgb")
    pix_rgb_hsv = np.flip(matplotlib.colors.rgb_to_hsv(pix_rgb).reshape((-1, 3)), 1)
    # pix_rgb_hsv = np.roll(matplotlib.colors.rgb_to_hsv(pix_rgb).reshape((-1, 3)), 1, axis=1)
    # pix_rgb_hsv = matplotlib.colors.rgb_to_hsv(pix_rgb)

    img_rgb = Image.fromarray(pix_rgb)
    # img_rgb.show()
    # sys.exit()

    print("sort pix_orig_hsv_view")
    pix_orig_hsv_view = pix_orig_hsv.reshape((-1, )).view("f4,f4,f4")
    idx_orig = np.argsort(pix_orig_hsv_view)
    pix_orig_sort = pix_orig.reshape((-1, 3))[idx_orig].reshape(pix_orig.shape)
    img_orig_sort = Image.fromarray(pix_orig_sort)
    # img_orig_sort.show()

    print("sort pix_rgb_hsv_view")
    pix_rgb_hsv_view = pix_rgb_hsv.reshape((-1, )).view("f4,f4,f4")
    idx_rgb = np.argsort(pix_rgb_hsv_view)
    pix_rgb_sort = pix_rgb.reshape((-1, 3))[idx_rgb].reshape(pix_rgb.shape)
    img_rgb_sort = Image.fromarray(pix_rgb_sort)
    # img_rgb_sort.show()

    pix_rgb_new = np.empty(pix_rgb.shape, dtype=np.uint8).reshape((-1, 3))
    pix_rgb_new[idx_orig] = pix_rgb.reshape((-1, 3))[idx_rgb]
    pix_rgb_new = pix_rgb_new.reshape(pix_orig.shape)
    img_rgb_new = Image.fromarray(pix_rgb_new)
    img_rgb_new.show()

    img_rgb_new.save(choosen_img_path.replace(".png", "_approx_nr_1.png"))


def create_mosaic_images():
    path_datas = "datas/"

    if not os.path.exists(path_datas):
        os.makedirs(path_datas)

    file_path = path_datas+"mosaic_pixs_nr_14.pkl.gz"

    if not os.path.exists(file_path):
        n = 5000
        height = 10
        width = 10
        mosaic_pixs = np.empty((n, height, width, 3), dtype=np.uint8)
        mosaic_pixs_hsv = np.empty((n, height, width, 3), dtype=np.float32)
        for i in range(0, n):
            if (i+1) % 1000 == 0:
                print("i: {}".format(i+1))

            # height = int(np.random.randint(30, 101))
            # width = int(np.random.randint(30, 101))
            pix = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

            pix_hsv = matplotlib.colors.rgb_to_hsv(pix)
            pix_hsv_view = np.flip(pix_hsv.reshape((-1, 3)), 1).reshape((-1, )).view("f4,f4,f4")
            idx_sort = np.argsort(pix_hsv_view)
            pix_sort = pix.reshape((-1, 3))[idx_sort].reshape((height, width, 3))

            # print("pix.shape: {}".format(pix.shape))
            # img = Image.fromarray(pix)
            # img = img.resize((img.width*5, img.height*5))
            # img.show()
            
            # print("pix_sort.shape: {}".format(pix_sort.shape))
            # img = Image.fromarray(pix_sort)
            # img = img.resize((img.width*5, img.height*5))
            # img.show()

            mosaic_pixs[i] = pix
            mosaic_pixs_hsv[i] = matplotlib.colors.rgb_to_hsv(pix)
            # mosaic_pixs[i] = pix_sort

        # mosaic_pixs = np.array(mosaic_pixs, dtype=np.uint8)

        with gzip.open(file_path, "wb") as f:
            d = {"mosaic_pixs": mosaic_pixs, "mosaic_pixs_hsv": mosaic_pixs_hsv}
            dill.dump(d, f)
    else:
        with gzip.open(file_path, "rb") as f:
            d = dill.load(f)
            mosaic_pixs = d["mosaic_pixs"]
            mosaic_pixs_hsv = d["mosaic_pixs_hsv"]
    # sys.exit(-1)
    return mosaic_pixs, mosaic_pixs_hsv


def find_best_fitting_mosaic_images_for_image():
    mosaic_pixs, mosaic_pixs_hsv = create_mosaic_images()

    print("mosaic_pixs.shape: {}".format(mosaic_pixs.shape))
    print("mosaic_pixs_hsv.shape: {}".format(mosaic_pixs_hsv.shape))

    # sys.exit(-1)

    images_amount = 4
    # image_path_template = "images/img_4096_4096_nr_{num}.{extension}"
    image_path_template = "images/img_400_300_nr_{num}.{extension}"

    img_paths = []

    # if not os.path.exists("images/"):
    #     os.makedirs("images/")

    # for i in range(1, images_amount+1):
    #     image_path = image_path_template.format(num=i, extension="png")
    #     if os.path.exists(image_path):
    #         img_paths.append(image_path)
    #         continue
        
    #     image_path_jpg = image_path_template.format(num=i, extension="jpg")
    #     if os.path.exists(image_path_jpg):
    #         img = Image.open(image_path_jpg)
    #         image_path_png = image_path_template.format(num=i, extension="png")
    #         img.save(image_path_png)
    #         continue

    # print("img_paths:\n{}".format(img_paths))

    # if len(img_paths) == 0:
    #     return

    img_paths = ["images/my_face.jpg"]

    choosen_img_path = img_paths[0]
    img_orig = Image.open(choosen_img_path)
    choosen_img_path = choosen_img_path.replace(".jpg", ".png")
    
    img_orig = img_orig.resize((img_orig.width//4, img_orig.height//4))
    pix_orig = np.array(img_orig)
    img_orig.show()

    # img_orig = img_orig.resize((img_orig.width*2, img_orig.height*2))
    # pix_orig = np.array(img_orig)
    # img_orig2 = img_orig.resize((img_orig.width//4, img_orig.height//4))
    # img_orig2.show()

    # pix_part = pix_orig[100:200, 150:280]
    # pix_rgb = np.random.randint(0, 256, pix_part.shape, dtype=np.uint8)

    # pix_orig_hsv = matplotlib.colors.rgb_to_hsv(pix_orig)
    # pix_rgb_hsv = matplotlib.colors.rgb_to_hsv(pix_rgb)

    # mean_h_orig = np.mean(pix_orig_hsv[:, :, 0])
    # mean_h_rgb = np.mean(pix_rgb_hsv[:, :, 0])

    # print("mean_h_orig: {}".format(mean_h_orig))
    # print("mean_h_rgb: {}".format(mean_h_rgb))

    # pix_rgb_hsv[:, :, 0] = pix_rgb_hsv[:, :, 0] * mean_h_orig/mean_h_rgb

    # pix_rgb_conv = matplotlib.colors.hsv_to_rgb(pix_rgb_hsv).astype(np.uint8)

    # print("pix_rgb_conv.shape: {}".format(pix_rgb_conv.shape))
    # print("pix_rgb_conv.dtype: {}".format(pix_rgb_conv.dtype))

    # img_part = Image.fromarray(pix_part)
    # img_rgb = Image.fromarray(pix_rgb)
    # img_rgb_conv = Image.fromarray(pix_rgb_conv)

    # img_part.show()
    # img_rgb.show()
    # img_rgb_conv.show()

    # sys.exit(-1)

    pix_copy = pix_orig.copy()

    h, w, channels = pix_orig.shape

    h_m, w_m, channels_m = mosaic_pixs[0].shape
    print("h_m: {}, w_m: {}".format(h_m, w_m))

    h_parts = h//h_m
    w_parts = w//w_m

    chosen_hsv_channel = 2
    h_means = np.sum(np.sum(mosaic_pixs_hsv[:, :, :, 0], axis=-1), axis=-1) / (h_m*w_m)
    s_means = np.sum(np.sum(mosaic_pixs_hsv[:, :, :, 1], axis=-1), axis=-1) / (h_m*w_m)
    v_means = np.sum(np.sum(mosaic_pixs_hsv[:, :, :, 2], axis=-1), axis=-1) / (h_m*w_m)

    for i in range(0, 6000):
    # for j in range(0, h_parts):
      # print("j: {}".format(j))
      # for i in range(0, w_parts):
        # print("j: {}, i: {}".format(j, i))
        # y = j*h_m
        # x = i*w_m
        y = int(np.random.randint(0, h-h_m))
        x = int(np.random.randint(0, w-w_m))

        if (i+1) % 100 == 0:
            print("i: {}".format(i+1))
        # print("y: {}, x: {}".format(y, x))

        pix_choosen = pix_orig[y:y+h_m, x:x+w_m]
        # img_choosen = Image.fromarray(pix_choosen)
        # img_choosen.show()

        # pix_part = pix_orig[100:200, 150:280]
        pix_choosen_hsv = matplotlib.colors.rgb_to_hsv(pix_choosen)
        h_mean = np.mean(pix_choosen_hsv[:, :, 0])
        s_mean = np.mean(pix_choosen_hsv[:, :, 1])
        v_mean = np.mean(pix_choosen_hsv[:, :, 2])
        # pix_rgb = np.random.randint(0, 256, pix_part.shape, dtype=np.uint8)

        # pix_orig_hsv = matplotlib.colors.rgb_to_hsv(pix_orig)
        # pix_rgb_hsv = matplotlib.colors.rgb_to_hsv(pix_rgb)

        # mean_h_orig = np.mean(pix_orig_hsv[:, :, 0])
        # mean_h_rgb = np.mean(pix_rgb_hsv[:, :, 0])

        diff_sums = np.sum(np.abs(mosaic_pixs.astype(np.double)-pix_choosen.astype(np.double)).reshape((mosaic_pixs.shape[0], -1)), axis=-1)
        # diff_sums = np.sum(np.sum(np.sum(np.abs(mosaic_pixs.astype(np.double)-pix_choosen.astype(np.double)), axis=-1), axis=-1), axis=-1)

        idx_min = np.argmin(diff_sums)

        # idx_min_h = np.argmin(np.abs(h_means-h_mean))
        # idx_min_v = np.argmin(np.abs(v_means-v_mean))

        pix_rgb_hsv = mosaic_pixs_hsv[idx_min]
        # pix_rgb_hsv = mosaic_pixs_hsv[idx_min_h]
        h_mean_mosaic = np.mean(pix_rgb_hsv[:, :, 0])
        s_mean_mosaic = np.mean(pix_rgb_hsv[:, :, 1])
        v_mean_mosaic = np.mean(pix_rgb_hsv[:, :, 2])

        # pix_rgb = mosaic_pixs[idx_min_h]
        # pix_rgb_hsv = matplotlib.colors.rgb_to_hsv(pix_rgb)

        if h_mean_mosaic > 0.001:
            pix_rgb_hsv[:, :, 0] *= h_mean/h_mean_mosaic
        pix_rgb_hsv[:, :, 1] = 0.00
        # if (s_mean_mosaic > 0.001 or s_mean > 0.005) and s_mean/s_mean_mosaic > 0.001:
        #     pix_rgb_hsv[:, :, 1] *= s_mean/s_mean_mosaic
        if (v_mean_mosaic > 0.1 or v_mean > 0.5) and v_mean/v_mean_mosaic > 0.01:
            pix_rgb_hsv[:, :, 2] *= v_mean/v_mean_mosaic
        # pix_rgb_hsv[:, :, chosen_hsv_channel] *= h_mean/h_mean_mosaic

        pix_rgb_conv = matplotlib.colors.hsv_to_rgb(pix_rgb_hsv).astype(np.uint8)

        # diffs = mosaic_pixs.astype(np.double)-pix_choosen.astype(np.double)
        # print("diffs.shape: {}".format(diffs.shape))

        # diffs_sum = np.sum(np.sum(np.sum(np.abs(diffs), axis=-1), axis=-1), axis=-1)
        # # print("diffs_sum: {}".format(diffs_sum))

        # idx_min = np.argmin(diffs_sum)
        # # print("i: {}, idx_min: {}".format(i, idx_min))

        # pix_choosen[:] = mosaic_pixs[idx_min]
        pix_copy[y:y+h_m, x:x+w_m] = pix_rgb_conv
        # pix_copy[y:y+h_m, x:x+w_m] = mosaic_pixs[idx_min]

    img_copy = Image.fromarray(pix_copy)
    img_copy.show()
    # img_copy2 = img_copy.resize((img_copy.width//4, img_copy.height//4))
    # img_copy2.show()
    img_copy.save(choosen_img_path.replace(".png", "_approx_nr_1.png"))


    # try different approaches for random pixels, e.g. take average of each row or col and create the new pix_rgb!
    
    # take n different pixels from pix_orig and do np.tile until you get w x h image again
    # h, w, channels = pix_orig.shape
    # n = 16
    # colors_choosen = pix_orig.reshape((-1, channels))[np.random.permutation(np.arange(0, h*w))[:n]]
    # # pix_rgb = np.repeat(colors_choosen.reshape((-1, )), ((h*w)//n+1)*n).reshape((-1, 3))[:h*w*channels].reshape(pix_orig.shape)
    # pix_rgb = np.tile(colors_choosen.reshape((-1, )), (h*w)//n+1)[:h*w*channels].reshape(pix_orig.shape)

    # mean of each col and create a full new image of cols mean value pixels!
    # cols_mean = np.mean(pix_orig.astype(np.double), axis=0).astype(np.uint8)
    # print("cols_mean.shape: {}".format(cols_mean.shape))
    # pix_rgb = np.tile(cols_mean.reshape((-1, )), pix_orig.shape[0]).reshape(pix_orig.shape)

    # mean of each row and create a full new image of rows mean value pixels!
    # cols_mean = np.mean(pix_orig.astype(np.double), axis=1).astype(np.uint8)
    # print("cols_mean.shape: {}".format(cols_mean.shape))
    # pix_rgb = np.tile(cols_mean.reshape((-1, )), pix_orig.shape[1]).reshape(pix_orig.shape)

    # random pixels
    # pix_rgb = np.random.randint(0, 256, pix_orig.shape).astype(np.uint8)

    # print("convert from rgb to hsv for pix_orig")
    # pix_orig_hsv = np.flip(matplotlib.colors.rgb_to_hsv(pix_orig).reshape((-1, 3)), 1)
    # # pix_orig_hsv = np.roll(matplotlib.colors.rgb_to_hsv(pix_orig).reshape((-1, 3)), 1, axis=1)
    # # pix_orig_hsv = matplotlib.colors.rgb_to_hsv(pix_orig)

    # print("convert from rgb to hsv for pix_rgb")
    # pix_rgb_hsv = np.flip(matplotlib.colors.rgb_to_hsv(pix_rgb).reshape((-1, 3)), 1)
    # # pix_rgb_hsv = np.roll(matplotlib.colors.rgb_to_hsv(pix_rgb).reshape((-1, 3)), 1, axis=1)
    # # pix_rgb_hsv = matplotlib.colors.rgb_to_hsv(pix_rgb)

    # img_rgb = Image.fromarray(pix_rgb)
    # # img_rgb.show()
    # # sys.exit()

    # print("sort pix_orig_hsv_view")
    # pix_orig_hsv_view = pix_orig_hsv.reshape((-1, )).view("f4,f4,f4")
    # idx_orig = np.argsort(pix_orig_hsv_view)
    # pix_orig_sort = pix_orig.reshape((-1, 3))[idx_orig].reshape(pix_orig.shape)
    # img_orig_sort = Image.fromarray(pix_orig_sort)
    # # img_orig_sort.show()

    # print("sort pix_rgb_hsv_view")
    # pix_rgb_hsv_view = pix_rgb_hsv.reshape((-1, )).view("f4,f4,f4")
    # idx_rgb = np.argsort(pix_rgb_hsv_view)
    # pix_rgb_sort = pix_rgb.reshape((-1, 3))[idx_rgb].reshape(pix_rgb.shape)
    # img_rgb_sort = Image.fromarray(pix_rgb_sort)
    # # img_rgb_sort.show()

    # pix_rgb_new = np.empty(pix_rgb.shape, dtype=np.uint8).reshape((-1, 3))
    # pix_rgb_new[idx_orig] = pix_rgb.reshape((-1, 3))[idx_rgb]
    # pix_rgb_new = pix_rgb_new.reshape(pix_orig.shape)
    # img_rgb_new = Image.fromarray(pix_rgb_new)
    # img_rgb_new.show()

    # img_rgb_new = Image.fromarray(pix_copy)
    # img_rgb_new.save(choosen_img_path.replace(".png", "_approx_nr_1.png"))


def create_from_image_mosaic_image(source_path):
    def get_finished_datas_pixses(datas_file_path):
        with gzip.open(datas_file_path, "rb") as f:
            datas = dill.load(f)
        return datas


    def get_argsort_table(pixses_src_rgb, pixses_rgb, prefix=""):
        n = pixses_rgb.shape[0]

        pix_src_f = pix_src.astype(np.double)
        pixses_rgb_f = pixses_rgb.astype(np.double)

        argsort_table = np.empty((pixses_src_rgb.shape[0], n), dtype=np.uint32)

        for idx, pix_ in enumerate(pixses_src_rgb, 0):
            print("{}idx: {}".format(prefix, idx))

            pix_ = pix_.astype(np.double)

            # find best fitting pixses_rgb first!
            n_parts = 3
            n_part_len = n//n_parts+1
            sums = np.array([])
            for i_part in range(0, n_parts):
                pixses_rgb_f_part = pixses_rgb_f[n_part_len*i_part:n_part_len*(i_part+1)]
                sums_part = np.sum(np.abs(pix_.reshape((1, )+pix_.shape)-pixses_rgb_f_part).reshape((pixses_rgb_f_part.shape[0], -1)), axis=-1)
                sums = np.hstack((sums, sums_part))
            arg_sort = np.argsort(sums)
            argsort_table[idx] = arg_sort
            print("{} - arg_sort[:10]: {}".format(prefix, arg_sort[:10]))

        return argsort_table


    datas_folder = ROOT_PATH+"datas/"
    if not os.path.exists(datas_folder):
        os.makedirs(datas_folder)

    tile_w, tile_h = 60, 45
    print("tile_h: {}".format(tile_h))
    print("tile_w: {}".format(tile_w))
    datas_file_path_template = datas_folder+"all_pixabay_com_pixses_{h}_{w}_part_{{:02}}.pkl.gz".format(h=tile_h, w=tile_w)
    amount_pkl_files = 20
    pixses_rgb_lst = [get_finished_datas_pixses(datas_file_path_template.format(i)) for i in range(0, amount_pkl_files)]
    pixses_rgb_orig = np.vstack(pixses_rgb_lst)

    print("pixses_rgb_orig.shape: {}".format(pixses_rgb_orig.shape))

    t2 = 15
    tile_w_new, tile_h_new = 4*t2, 3*t2
    datas_file_pixses_smaller = "datas/all_pixabay_com_pixses_{h}_{w}.pkl.gz".format(h=tile_h_new, w=tile_w_new)

    pixses_rgb_smaller = np.empty((pixses_rgb_orig.shape[0], tile_h_new, tile_w_new, 3), dtype=np.uint8)
    for i, pix in enumerate(pixses_rgb_orig, 0):
        if i%1000 == 0:
            print("i: {}".format(i))
        pixses_rgb_smaller[i] = np.array(Image.fromarray(pix).resize((tile_w_new, tile_h_new), Image.LANCZOS))

    img_src = Image.open(source_path)
    pix_src = np.array(img_src)
    
    h, w = pix_src.shape[:2]
    h1, w1 = pixses_rgb_smaller[0].shape[:2]

    if h % h1 != 0 or w % w1 != 0:
        pix_src = pix_src[:(h//h1)*h1, :(w//w1)*w1]

    if pix_src.shape[2]  == 4:
        pix_src = pix_src[:, :, :3]

    destination_path = source_path.replace(".png", "_tw_{tw}_th_{th}_mosaic.png".format(tw=tile_w_new, th=tile_h_new))

    if not os.path.exists(destination_path):
        pix_dst = pix_src.copy()
        img_dst = Image.fromarray(pix_dst)
        img_dst.save(destination_path)
    else:
        img_dst = Image.open(destination_path)
        pix_dst = np.array(img_dst)

    x_min = 0
    x_max = w//w1

    y_min = 0
    y_max = h//h1

    print("x_min: {}, x_max: {}".format(x_min, x_max))
    print("y_min: {}, y_max: {}".format(y_min, y_max))

    ys = np.tile(np.arange(0, y_max), x_max).reshape((x_max, y_max)).T.reshape((-1, ))
    xs = np.tile(np.arange(0, x_max), y_max)
    indexes = np.vstack((ys, xs)).T

    idxs = np.dstack((ys.reshape((y_max, x_max)), xs.reshape((y_max, x_max))))
    idxs2 = np.dstack((idxs, idxs+1))

    idxs2_1 = idxs2.copy().reshape((-1, 4))*(tile_h_new, tile_w_new, tile_h_new, tile_w_new)
    idxs2_2 = idxs2[:-1].copy().reshape((-1, 4))*(tile_h_new, tile_w_new, tile_h_new, tile_w_new)+(tile_h_new//2, 0, tile_h_new//2, 0)
    idxs2_3 = idxs2[:, :-1].copy().reshape((-1, 4))*(tile_h_new, tile_w_new, tile_h_new, tile_w_new)+(0, tile_w_new//2, 0, tile_w_new//2)
    idxs2_4 = idxs2[:-1, :-1].copy().reshape((-1, 4))*(tile_h_new, tile_w_new, tile_h_new, tile_w_new)+(tile_h_new//2, tile_w_new//2, tile_h_new//2, tile_w_new//2)
    choosen_indexes_all = np.vstack((idxs2_1, idxs2_2, idxs2_3, idxs2_4))


    for doing_parts_number in range(0, 1):
        print("doing_parts_number: {}".format(doing_parts_number))
        idxs_rnd = np.random.permutation(np.arange(0, pixses_rgb_smaller.shape[0]))[:50000]
        pixses_rgb = pixses_rgb_smaller[idxs_rnd]

        print("pixses_rgb.shape: {}".format(pixses_rgb.shape))

        choosen_idx = np.random.permutation(np.arange(0, choosen_indexes_all.shape[0])) # [:1000]
        choosen_idx = choosen_idx[:choosen_idx.shape[0]//6]
        choosen_indexes = choosen_indexes_all[choosen_idx]
        print("choosen_idx: {}".format(choosen_idx))
        print("choosen_indexes.shape: {}".format(choosen_indexes.shape))

        pixses_src_rgb = np.empty((choosen_indexes.shape[0], h1, w1, 3), dtype=np.uint8)
        for i, (y1, x1, y2, x2) in enumerate(choosen_indexes, 0):
            pix_ = pix_src[y1:y2, x1:x2]
            pixses_src_rgb[i] = pix_

        idxs_y = np.zeros((h1, w1), dtype=np.int64)+np.arange(0, h1).reshape((-1, 1)).astype(np.int64)
        idxs_x = np.zeros((h1, w1), dtype=np.int64)+np.arange(0, w1).reshape((1, -1)).astype(np.int64)

        idxs = np.dstack((idxs_y, idxs_x))
        print("idxs.shape: {}".format(idxs.shape))

        h2 = h1
        w2 = w1
        print("h2: {}".format(h2))
        print("w2: {}".format(w2))
        idxs_parts = ( idxs
            .reshape((h2//t2, t2, w2, 2))
            .transpose(0, 2, 1, 3)
            .reshape((h2//t2*w2//t2, t2, t2, 2))
            .transpose(0, 2, 1, 3)
        )

        idxs_1d_y, idxs_1d_x = idxs_parts.reshape((-1, 2)).T

        def get_feature_matrix(pixses):    
            m_row_sum_feature = np.sum(pixses, axis=1).transpose(0, 2, 1).reshape((pixses.shape[0], -1))
            m_col_sum_feature = np.sum(pixses, axis=2).transpose(0, 2, 1).reshape((pixses.shape[0], -1))
            m_5x5_feature = np.sum(pixses[:, idxs_1d_y, idxs_1d_x].reshape((pixses.shape[0], -1, t2*t2, 3)), axis=2).transpose(0, 2, 1).reshape((pixses.shape[0], -1))

            m_feature = np.hstack((m_row_sum_feature, m_col_sum_feature, m_5x5_feature))

            return m_feature

        print("Calculating 'm_rgb_features'!")
        m_rgb_features = get_feature_matrix(pixses_rgb)
        print("Calculating 'm_src_features'!")
        m_src_features = get_feature_matrix(pixses_src_rgb)

        print("pixses_rgb.nbytes: {}".format(pixses_rgb.nbytes))
        
        print("pixses_src_rgb.shape: {}".format(pixses_src_rgb.shape))
        print("pixses_rgb.shape: {}".format(pixses_rgb.shape))
        
        cpu_amount = multiprocessing.cpu_count()

        n_src = pixses_src_rgb.shape[0]
        idx_parts = np.arange(0, cpu_amount+1)*(n_src//cpu_amount+1)

        print("idx_parts: {}".format(idx_parts))

        def process(proc_num, receiver, sender):
            pixses_src_rgb_part = receiver.recv()
            pixses_rgb = receiver.recv()
            argsort_table = get_argsort_table(pixses_src_rgb_part, pixses_rgb, prefix="{}: ".format(proc_num))
            sender.send(argsort_table)

        m_src_features_parts = [m_src_features[idx1:idx2] for idx1, idx2 in zip(idx_parts[:-1], idx_parts[1:])]
        pipes_proc_main = [Pipe() for _ in range(cpu_amount)]
        pipes_main_proc = [Pipe() for _ in range(cpu_amount)]
        receivers_proc, senders_main = list(zip(*pipes_proc_main))
        receivers_main, senders_proc = list(zip(*pipes_main_proc))

        procs = [Process(target=process, args=(proc_num, receiver, sender)) for proc_num, (receiver, sender) in enumerate(zip(receivers_proc, senders_proc))]
        for proc in procs:
            proc.start()

        for idx, (m_src_features_part, sender) in enumerate(zip(m_src_features_parts, senders_main), 0):
            print("send to proc: idx: {}".format(idx))
            sender.send(m_src_features_part)
            sender.send(m_rgb_features)

        argsort_table = np.zeros((0, pixses_rgb.shape[0]), dtype=np.uint32)
        for idx, receiver in enumerate(receivers_main, 0):
            print("receive from proc: idx: {}".format(idx))
            argsort_table_part = receiver.recv()
            argsort_table = np.vstack((argsort_table, argsort_table_part))

        for proc in procs:
            proc.join()

        used_idx = {}
        for i, (y1, x1, y2, x2) in enumerate(choosen_indexes, 0):
            arg_sort = argsort_table[i]
            for i, idx in enumerate(arg_sort, 0):
                if not idx in used_idx:
                    used_idx[idx] = 0
                    break

            print("i: {}, idx: {}".format(i, idx))

            pix_choosen = pixses_rgb[idx]
            pix_dst[y1:y2, x1:x2] = pix_choosen


        img_dst = Image.fromarray(pix_dst)
        img_dst.save(destination_path)
        img_dst.save(destination_path.replace(".png", ".jpg"))

    print("(tile_w_new, tile_h_new): {}".format((tile_w_new, tile_h_new)))


if __name__ == "__main__":
    # random_256_256_pallete()

    # random_4096_4096_pallete()
    # approx_image_with_all_rgb()

    # find_best_fitting_image_for_random_pixels()
    # find_best_fitting_mosaic_images_for_image()
    

    def get_pixses_dict_object(mosaic_pixs_dir, obj_file_path):
        if not os.path.exists(obj_file_path):
            root_path_dir, dir_names, file_names = next(os.walk(mosaic_pixs_dir))

            img_ = Image.open(root_path_dir+file_names[0])
            pix_ = np.array(img_)
            shape_ = pix_.shape
            h1 = shape_[0]
            w1 = shape_[1]

            # print("h1: {}, w1: {}".format(h1, w1))
            # print("mosaic_pixs_dir: {}".format(mosaic_pixs_dir))
            # print("obj_file_path: {}".format(obj_file_path))
            # sys.exit(-1)

            pixses_dict = {}
            for it, file_name in enumerate(file_names[:100], 0):
                print("it: {}".format(it))
                img = Image.open(root_path_dir+file_name)
                pix = np.array(img)
                if len(pix.shape) == 2:
                    pix = np.tile(pix.reshape((-1, )), 3).reshape((3, -1)).T.reshape((h1, w1, 3))
                elif pix.shape[2] == 4:
                    pix = pix[:, :, :3]
                pixses_dict[file_name] = pix

            with gzip.open(obj_file_path, "wb") as f:
                dill.dump(pixses_dict, f)
        else:
            with gzip.open(obj_file_path, "rb") as f:
                pixses_dict = dill.load(f)

        need_again_save = False
        for key, val in pixses_dict.items():
            if val.shape[2] == 4:
                pixses_dict[key] = val[:, :, :3]
                need_again_save = True

        if need_again_save:
            with gzip.open(obj_file_path, "wb") as f:
                dill.dump(pixses_dict, f)

        return pixses_dict


    # mosaic_pixs_dir = ROOT_PATH+"images/pixabay_com/pngs/40x30/"
    # mosaic_pixs_dir = ROOT_PATH+"images/pixabay_com/pngs/20x15/"
    # obj_file_path = ROOT_PATH+"datas/pixabay_com_40x30_2.pkl.gz"
    # obj_file_path = ROOT_PATH+"images/pixabay_com/picture_dict_objs/60x45.pkl.gz"

    # TODO: create pixses_dict bigger!!!
    # with gzip.open(obj_file_path, "rb") as f:
    #     pixses_dict = dill.load(f)
    # pixses_dict = get_pixses_dict_object(mosaic_pixs_dir, obj_file_path)

    # print("len(pixses_dict): {}".format(len(pixses_dict)))
    # print("obj_file_path: {}".format(obj_file_path))


    argv = sys.argv
    if len(argv) < 3:
        print("need second argument for the path of the image!")
        print("need third argument for the resize of image!")
        sys.exit(-1)

    source_path_orig = argv[1]
    if not os.path.exists(source_path_orig):
        print("File '{}' does not exists!".format(source_path_orig))
        print("Exit program!")
        sys.exit(-2)

    try:
        resize_factor = int(argv[2])
        if resize_factor < 1:
            resize_factor = 1
        elif resize_factor > 10:
            resize_factor = 10
    except:
        print("resize_factor cannot convert to int!")
        print("Exit program!")
        sys.exit(-3)

    print("resize_factor set to: {}".format(resize_factor))

    img_src = Image.open(source_path_orig)
    pix_src = np.array(img_src)

    if len(pix_src.shape) == 2:
        pix_src = np.dstack((pix_src, pix_src, pix_src))
        img_src = Image.fromarray(pix_src)
    if pix_src.shape[2] == 4:
        pix_src = pix_src[:, :, :3]
        img_src = Image.fromarray(pix_src)

    extensions = ['.jpeg', '.jpg']
    for extension in extensions:
        if extension in source_path_orig:
            break
    extension_used = extension
    # extension_used = '.jpg'
    source_path = source_path_orig.replace(extension_used, "_changed_size_{}.png".format(resize_factor))

    if not os.path.exists(source_path):
        img_src = img_src.resize((img_src.width*resize_factor, img_src.height*resize_factor), Image.LANCZOS)
        img_src.save(source_path)

    create_from_image_mosaic_image(source_path)
