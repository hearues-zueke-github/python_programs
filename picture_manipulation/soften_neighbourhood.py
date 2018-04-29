#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from os.path import expanduser

def get_2d_idx_table(m, n):
    idx_2d_table = np.zeros((m, n, 2)).astype(np.int)

    idx_2d_table[:, :, 0] += np.arange(0, m).reshape((m, -1))
    idx_2d_table[:, :, 1] += np.arange(0, n).reshape((-1, n))

    return idx_2d_table

def apply_kernel_on_picture(pix, weight_vector):
    h, w, d = pix.shape

    idx_frame_3x3 = get_2d_idx_table(3, 3).reshape((-1, 2))
    idx_frame_hxw = get_2d_idx_table(h, w).reshape((-1, 2))

    val_table_ext_row = np.vstack((pix[0].reshape((1, -1, d)),
                                   pix[:],
                                   pix[-1].reshape((1, -1, d))))
    
    val_table_ext = np.hstack((val_table_ext_row[:, 0].reshape((-1, 1, d)),
                               val_table_ext_row[:],
                               val_table_ext_row[:, -1].reshape((-1, 1, d))))

    idx_table_full = np.vstack([idx_frame_3x3+idx for idx in idx_frame_hxw])
    print("idx_table_full.shape:\n{}".format(idx_table_full.shape))

    vals_table_full = val_table_ext[idx_table_full.T.tolist()]
    vals_table_full = vals_table_full.reshape((-1, 9, 3))

    print("vals_table_full.shape: {}".format(vals_table_full.shape))
    mean_val_table = np.sum(vals_table_full.astype(np.float)*weight_vector, axis=1)
    mean_val_table[mean_val_table < 0.] = 0.
    mean_val_table[mean_val_table >255.] = 255.
    mean_val_table = mean_val_table.reshape((h, w, d))

    return mean_val_table.astype(np.uint8)

def apply_kernel_on_picture_2(pix, weight_array):
    assert isinstance(pix, np.ndarray)
    assert isinstance(weight_array, np.ndarray)

    assert pix.ndim == 3
    assert weight_array.ndim == 2

    h, w, d = pix.shape
    h1, w1 = weight_array.shape

    append_t = h1//2
    append_d = (h1-1)//2
    append_l = w1//2
    append_r = (w1-1)//2

    weight_vector = weight_array.reshape((-1, 1))

    row_0 = pix[0].reshape((1, -1, d))
    for _ in xrange(1, append_t):
        row_0 = np.vstack((row_0, pix[0].reshape((1, -1, d))))

    row_1 = pix[-1].reshape((1, -1, d))
    for _ in xrange(1, append_d):
        row_1 = np.vstack((row_1, pix[-1].reshape((1, -1, d))))

    # print("row_0.shape: {}".format(row_0.shape))
    # print("pix.shape: {}".format(pix.shape))
    # print("row_1.shape: {}".format(row_1.shape))
    val_table_ext_row = np.vstack((row_0,
                                   pix[:],
                                   row_1))

    col_0 = val_table_ext_row[:, 0].reshape((-1, 1, d))
    for _ in xrange(1, append_l):
        col_0 = np.hstack((col_0, val_table_ext_row[:, 0].reshape((-1, 1, d))))

    col_1 = val_table_ext_row[:, -1].reshape((-1, 1, d))
    for _ in xrange(1, append_r):
        col_1 = np.hstack((col_1, val_table_ext_row[:, -1].reshape((-1, 1, d))))

    val_table_ext = np.hstack((col_0,
                               val_table_ext_row[:],
                               col_1))

    # print("val_table_ext.shape: {}".format(val_table_ext.shape))

    pix_all = np.zeros((h, w, 0)).astype(np.float)
    # print("pix_all.shape: {}".format(pix_all.shape))

    for j in xrange(0, h1):
        for i in xrange(0, w1):
            pix_all = np.dstack((pix_all, val_table_ext[j:h+j, i:w+i]))

    pix_all = pix_all.reshape((h, w, -1, d))

    new_pix = np.sum(pix_all*weight_vector, axis=2)
    new_pix[new_pix < 0.] = 0.
    new_pix[new_pix >255.] = 255.

    return new_pix.astype(np.uint8)

if __name__ == "__main__":
    home_path = expanduser("~")
    print("home_path: {}".format(home_path))

    path = home_path+"/Pictures/soften_neighbourhood"
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)

    orig_img = Image.open("nature-trees-640x480-wallpaper.jpg")
    pix = np.array(orig_img)
    print("pix.shape: {}".format(pix.shape))

    weight_vector_1 = np.ones((9, ))*0.05
    weight_vector_1[4] = 1.
    weight_vector_1 = weight_vector_1/np.sum(weight_vector_1)
    weight_vector_1 = weight_vector_1.reshape((9, 1))
    print("weight_vector_1: {}".format(weight_vector_1))

    weight_vector_2 = np.zeros((9, )).reshape((3, 3))
    weight_vector_2[:, 0] = -1.
    weight_vector_2[:, 2] = 1.
    weight_vector_2 = weight_vector_2.reshape((9, 1))
    print("weight_vector_2: {}".format(weight_vector_2))

    weight_vector_3 = np.zeros((9, )).reshape((3, 3))
    weight_vector_3[0] = -1.
    weight_vector_3[2] = 1.
    weight_vector_3 = weight_vector_3.reshape((9, 1))
    print("weight_vector_3: {}".format(weight_vector_3))

    # pix_1 = apply_kernel_on_picture(pix, weight_vector_1)
    # pix_2 = apply_kernel_on_picture(pix, weight_vector_2)
    # pix_3 = apply_kernel_on_picture(pix, weight_vector_3)

    print("Apply weight_vector_1 to pix")
    pix_new_1 = apply_kernel_on_picture_2(pix, weight_vector_1.reshape((3, 3)))
    print("Apply weight_vector_2 to pix")
    pix_new_2 = apply_kernel_on_picture_2(pix, weight_vector_2.reshape((3, 3)))
    print("Apply weight_vector_3 to pix")
    pix_new_3 = apply_kernel_on_picture_2(pix, weight_vector_3.reshape((3, 3)))

    img = Image.fromarray(pix)
    img.save("orig_img.png", "PNG")

    # img_1 = Image.fromarray(pix_1)
    # img_1.save("img_kernel_1.png", "PNG")

    # img_2 = Image.fromarray(pix_2)
    # img_2.save("img_kernel_2.png", "PNG")

    # img_3 = Image.fromarray(pix_3)
    # img_3.save("img_kernel_3.png", "PNG")

    print("Saving pix_new_1")
    img_new_1 = Image.fromarray(pix_new_1)
    img_new_1.save("img_kernel_new_1.png", "PNG")

    print("Saving pix_new_2")
    img_new_2 = Image.fromarray(pix_new_2)
    img_new_2.save("img_kernel_new_2.png", "PNG")

    print("Saving pix_new_3")
    img_new_3 = Image.fromarray(pix_new_3)
    img_new_3.save("img_kernel_new_3.png", "PNG")
