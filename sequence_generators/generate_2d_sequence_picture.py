#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import os
import sys
import time

import numpy as np

from indexed import IndexedOrderedDict

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

import utils_sequence

def create_x_y_arr(n):
    l = 2**n

    arr_y = np.zeros((l, l-1), dtype=np.int)
    arr_x = np.zeros((l-1, l), dtype=np.int)

    y = (l-1)//2**1
    w = l//2**0
    arr_x[y, w*0:w*1] = 1

    x = (l-1)//2**1
    h = l//2**1
    arr_y[h*0:h*1, x] = 1
    arr_y[h*1:h*2, x] = 2

    for iexp in range(1, n):
        y = (l-1)//2**(iexp+1)
        h = l//2**iexp
        w = l//2**iexp
        for iy in range(0, 2**iexp):
            for ix in range(0, 2**iexp):
                arr_x[y+h*iy, w*ix:w*(ix+1)] = (ix+iy+1)%2+1

        x = (l-1)//2**(iexp+1)
        w = l//2**iexp
        h = l//2**(iexp+1)
        for iy in range(0, 2**(iexp+1)):
            for ix in range(0, 2**iexp):
                arr_y[h*iy:h*(iy+1), x+w*ix] = (ix+iy+1)%2+1

    return arr_x, arr_y


def create_image_of_x_y_arr(n, arr_x, arr_y):
    l = 2**n

    pw = 1 # pixel width of one line
    pix = np.zeros((l*pw+l-1, l*pw+l-1, 3), dtype=np.uint8)

    col_red = (0xFF, 0x00, 0x00)
    col_green = (0x00, 0xFF, 0x00)
    col_black = (0x00, 0x00, 0x00)
    col_white = (0xFF, 0xFF, 0xFF)

    for y in range(pw, l*pw+l-1, pw+1):
        for x in range(pw, l*pw+l-1, pw+1):
            pix[y, x] = col_white

    for y in range(0, arr_y.shape[0]):
        for x in range(0, arr_y.shape[1]):
            v = arr_y[y, x]
            py = y*(pw+1)
            px = x*(pw+1)+pw
            if v == 1:
                pix[py:py+pw, px] = col_red
            elif v == 2:
                pix[py:py+pw, px] = col_green

    for y in range(0, arr_x.shape[0]):
        for x in range(0, arr_x.shape[1]):
            v = arr_x[y, x]
            py = y*(pw+1)+pw
            px = x*(pw+1)
            if v == 1:
                pix[py, px:px+pw] = col_red
            elif v == 2:
                pix[py, px:px+pw] = col_green

    return pix


def create_image_bw_of_x_y_arr(n, arr_x, arr_y):
    l = 2**n

    pix = np.zeros((l-1, l-1, 3), dtype=np.uint8)

    arr = (arr_x[:, :-1]-1)+(arr_x[:, 1:]-1)+(arr_y[:-1]-1)+(arr_y[1:]-1)
    pix[arr==3] = (0xFF, 0xFF, 0xFF) # .reshape((l-1, l-1, 1))
    return pix

def create_image_gray_of_x_y_arr(n, arr_x, arr_y):
    l = 2**n

    pix = np.zeros((l-1, l-1, 3), dtype=np.uint8)

    arr = (arr_x[:, :-1]-1)*2+(arr_x[:, 1:]-1)*8+(arr_y[:-1]-1)*1+(arr_y[1:]-1)*4
    arr = arr.reshape((l-1, l-1, 1))
    # idxs = arr<15
    # pix[idxs] = arr[idxs]*16
    # idxs = arr==15
    # pix[idxs] = arr[idxs]
    pix[:] = arr*16
    return pix


if __name__ == '__main__':
    # n = 2
    for n in range(1, 11):
        arr_x, arr_y = create_x_y_arr(n)
        pix = create_image_of_x_y_arr(n, arr_x, arr_y)
        pix_bw = create_image_bw_of_x_y_arr(n, arr_x, arr_y)
        pix_gray = create_image_gray_of_x_y_arr(n, arr_x, arr_y)
        
        resize_factor = 1
        img = Image.fromarray(pix)
        img = img.resize((img.height*resize_factor, img.width*resize_factor))
    
        resize_factor_2 = 3
        img_bw = Image.fromarray(pix_bw)
        img_bw = img_bw.resize((img_bw.height*resize_factor_2, img_bw.width*resize_factor_2))

        resize_factor_gray = 3
        img_gray = Image.fromarray(pix_gray)
        img_gray = img_gray.resize((img_gray.height*resize_factor_gray, img_gray.width*resize_factor_gray))


        path_img = 'images/'
        if not os.path.exists(path_img):
            os.makedirs(path_img)
        img.save(path_img+'2d_sequence_n_{}.png'.format(n))
        img_bw.save(path_img+'2d_sequence_bw_n_{}.png'.format(n))
        img_gray.save(path_img+'2d_sequence_gray_n_{}.png'.format(n))

        print("finished with n: {}".format(n))
