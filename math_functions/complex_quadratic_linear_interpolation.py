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

def get_factors(xs, ys):
    l = xs.shape[0]
    A = np.tile(xs, l).reshape((l, l)).T**np.arange(0, l, 1)
    A_inv = np.linalg.inv(A)
    asz = A_inv.dot(ys)
    return asz


def get_ys(xs, asz):
    l = asz.shape[0]
    Xs = np.tile(xs, l).reshape((l, -1)).T**np.arange(0, l)
    ys = Xs.dot(asz)
    return ys


def interpolate_xs_ys_points():
    n = 10
    xs = np.random.random((n, ))*10+0.5
    ys = np.random.random((n, ))*10+0.5

    plt.figure()
    plt.title("Complex numbers")
    plt.xlabel("real")
    plt.ylabel("imag")

    plt.plot(xs, ys, "b.")
    plt.plot(xs, ys, "g-")
    for x, y, s in zip(xs, ys, list(map(str, np.arange(0, n)))):
        plt.text(x+0.05, y+0.05, s)

    plt.close("all")

    plt.figure()
    plt.title("Complex numbers quadratic interpolations")
    plt.xlabel("real")
    plt.ylabel("imag")

    asz_lst = []

    k = 3 # amount of points for approximation, 3 points -> quadratic function

    # for creating a loop, the first k-1 points will be copied to the end
    xs = np.hstack((xs, xs[:k].copy()))
    ys = np.hstack((ys, ys[:k].copy()))
    print("xs: {}".format(xs))
    print("ys: {}".format(xs))

    ts = np.arange(0, k)
    ts_many = np.array([np.hstack((np.arange(t1, t2, (t2-t1)/100), (t2, ))) for t1, t2 in zip(ts[:-1], ts[1:])])
    ts_manys = []
    ts__ = np.array([ts[0]])
    for ts_ in ts_many:
        ts__ = np.hstack((ts__, ts_[1:]))
        ts_manys.append(ts__)

    for i in range(0, len(xs)-k+1):
        asz_x = get_factors(ts, xs[i:i+k])
        asz_y = get_factors(ts, ys[i:i+k])

        asz_lst.append((asz_x, asz_y))

        xs1 = get_ys(ts_manys[-1], asz_x)
        ys1 = get_ys(ts_manys[-1], asz_y)
        plt.plot(xs1, ys1, "g-")

    plt.plot(xs, ys, "b.")
    for x, y, s in zip(xs, ys, list(map(str, np.arange(0, n)))):
        plt.text(x+0.05, y+0.05, s)


    plt.figure()
    plt.title("Complex numbers the whole quadratic interpolation only")
    plt.xlabel("real")
    plt.ylabel("imag")

    plt.plot(xs, ys, "b.")
    for x, y, s in zip(xs, ys, list(map(str, np.arange(0, n)))):
        plt.text(x+0.05, y+0.05, s)

    asz_x_left, asz_y_left = asz_lst[0]
    # xs1 = get_ys(ts_many[0], asz_x_left)
    # ys1 = get_ys(ts_many[0], asz_y_left)

    # plt.plot(xs1, ys1, "r-")

    for i in range(0, len(xs)-k):
        xs1 = get_ys(ts_many[1], asz_x_left)
        ys1 = get_ys(ts_many[1], asz_y_left)

        # plt.plot(xs1, ys1, "g-")
        
        asz_x_right, asz_y_right = asz_lst[i+1]
        xs2 = get_ys(ts_many[0], asz_x_right)
        ys2 = get_ys(ts_many[0], asz_y_right)
        
        # plt.plot(xs2, ys2, "k-")

        xs12 = xs1*(1-ts_many[0])+xs2*ts_many[0]
        ys12 = ys1*(1-ts_many[0])+ys2*ts_many[0]
        plt.plot(xs12, ys12, "r-")

        asz_x_left = asz_x_right
        asz_y_left = asz_y_right

    # xs1 = get_ys(ts_many[1], asz_x_left)
    # ys1 = get_ys(ts_many[1], asz_y_left)

    # plt.plot(xs1, ys1, "r-")

    plt.show()


def interpolate_complex_points():
    n = 10
    xs = np.random.random((n, ))*10+0.5
    ys = np.random.random((n, ))*10+0.5

    cmplxs = np.empty((n, ), dtype=np.complex)
    cmplxs.real = xs
    cmplxs.imag = ys

    # def get_factors(xs, ys):
    #     l = xs.shape[0]
    #     A = np.tile(xs, l).reshape((l, l)).T**np.arange(0, l, 1)
    #     A_inv = np.linalg.inv(A)
    #     asz = A_inv.dot(ys)
    #     return asz

    # def get_ys(xs, asz):
    #     l = asz.shape[0]
    #     Xs = np.tile(xs, l).reshape((l, -1)).T**np.arange(0, l)
    #     ys = Xs.dot(asz)
    #     return ys


def get_pix_rgb(Z):
    rows, cols = Z.shape
    pix_hsv = np.empty((rows, cols, 3), dtype=np.double)

    pix_hsv[..., 1] = 1.

    vals_abs = np.abs(Z)
    v_min = 0.
    v_max = np.max(vals_abs)
    pix_hsv[..., 2] = ((vals_abs-v_min)/(v_max-v_min))*0.9+0.1

    vals_angle = np.angle(Z)
    pix_hsv[..., 0] = (vals_angle+3.14159265358979323)/(3.1415926535898*2)

    pix_rgb_f = matplotlib.colors.hsv_to_rgb(pix_hsv)
    pix_rgb = (pix_rgb_f*255.).astype(np.uint8)

    return pix_rgb


if __name__ == "__main__":
    # interpolate_xs_ys_points()

    # TODO: need a better structure!

    images_path = "images/"
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    file_name_template = images_path+"imgage_interpolating_nr_{:02}.png"

    file_name = 'Zs.pkl.gz'

    if os.path.exists(file_name):
        rows = 400
        cols = 600
        
        x_start = -6.0
        x_end = 6.0
        y_mid = 0.
        dh = ((x_end-x_start)/cols*rows)/2
        y_start = y_mid-dh
        y_end = y_mid+dh

        X1 = np.arange(0, cols)/cols*(x_end-x_start)+x_start
        Y1 = (np.arange(0, rows)/rows*(y_end-y_start)+y_start)[::-1]

        X = X1*np.complex(1, 0)
        Y = Y1*np.complex(0, 1)

        argmin_x = np.argmin(np.abs(X))
        argmin_y = np.argmin(np.abs(Y))

        with gzip.open(file_name, 'rb') as f:
            Zs = dill.load(f)

        k = 3
        exponents = np.arange(0, k)
        xs = np.arange(0, k)
        Xs = (np.tile(xs, k).reshape((k, k)).T**exponents)
        Xs_inv = np.linalg.inv(Xs).T

        Azs_lst = []
        for idx, (Z1, Z2, Z3) in enumerate(zip(Zs[:-2], Zs[1:-1], Zs[2:]), 0):
            print("idx: {}".format(idx))
            Azs = np.dstack((Z1, Z2, Z3)).dot(Xs_inv)
            Azs_lst.append(Azs)


        m_parts = 5
        xs_all = np.tile(np.arange(0, m_parts)/m_parts, k).reshape((k, -1))+np.arange(0, k).reshape((-1, 1))
        Xs_all = np.hstack((xs_all.reshape((-1, )), (k-1, ))).reshape((-1, 1))**exponents
        
        print("xs_all:\n{}".format(xs_all))
        print("Xs_all:\n{}".format(Xs_all))

        # xs1 = xs_all[0]
        # xs2 = xs_all[1]
        xs1 = Xs_all[m_parts*0:m_parts*1+1]
        xs2 = Xs_all[m_parts*1:m_parts*2+1]

        print("xs1:\n{}".format(xs1))
        print("xs2:\n{}".format(xs2))

        print("xs_all.shape: {}".format(xs_all.shape))
        print("Xs_all.shape: {}".format(Xs_all.shape))
        Azs_left = Azs_lst[0]
        Azs_right = Azs_lst[1]

        d_alpha = 1 / xs1.shape[0]
        for i, (x1, x2) in enumerate(zip(xs1, xs2), 0):
        # for i, (x1, x2) in enumerate(Xs_all):
            print("i: {}".format(i))
            Z = Azs_left.dot(x2)*(1-i*d_alpha)+Azs_right.dot(x1)*(i*d_alpha)
            pix = get_pix_rgb(Z)
            pix[:, argmin_x:argmin_x+1] = ~pix[:, argmin_x:argmin_x+1]
            pix[argmin_y:argmin_y+1, :] = ~pix[argmin_y:argmin_y+1, :]
            pix[argmin_y, argmin_x] = ~pix[argmin_y, argmin_x]

            img = Image.fromarray(pix)
            img.save(file_name_template.format(i))

        sys.exit(0)

    n = 10
    xs = np.random.random((n, ))*10+0.5
    ys = np.random.random((n, ))*10+0.5

    rows = 400
    cols = 600
    
    x_start = -6.0
    x_end = 6.0
    y_mid = 0.
    dh = ((x_end-x_start)/cols*rows)/2
    y_start = y_mid-dh
    y_end = y_mid+dh

    X1 = np.arange(0, cols)/cols*(x_end-x_start)+x_start
    Y1 = (np.arange(0, rows)/rows*(y_end-y_start)+y_start)[::-1]

    X = X1*np.complex(1, 0)
    Y = Y1*np.complex(0, 1)

    C = X+Y.reshape((-1, 1))
    print("Y1: {}".format(Y1))
    print("C: {}".format(C))

    def get_f(a, b):
        def f(c):
            return c*b+np.complex(c.imag+a*b+np.cos(c), c.real+a+b)**1.3 # c**2+c*a-complex(b, c.real+c.imag)*c
            # return c**2*a+c*b+np.complex(c.imag, c.real+a+b)**1.3 # c**2+c*a-complex(b, c.real+c.imag)*c
        return f

    # TODO: create a better interpolation class/function/etc. whatever
    images_path = "images/"
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    file_name_template = "imgage_nr_{:02}.png"

    argmin_x = np.argmin(np.abs(X))
    argmin_y = np.argmin(np.abs(Y))

    Zs = []
    image_num = 0
    for i in range(-10, 11, 1):
        f = get_f(i*0.25, 2)
        fv = np.vectorize(f)
        Z = fv(C)
        print("Z: {}".format(Z))

        pix_rgb = get_pix_rgb(Z)

        print("pix_rgb.shape: {}".format(pix_rgb.shape))
        pix_rgb[:, argmin_x:argmin_x+1] = ~pix_rgb[:, argmin_x:argmin_x+1]
        pix_rgb[argmin_y:argmin_y+1, :] = ~pix_rgb[argmin_y:argmin_y+1, :]
        pix_rgb[argmin_y, argmin_x] = ~pix_rgb[argmin_y, argmin_x]

        img = Image.fromarray(pix_rgb)
        # img.show()
        img.save(images_path+file_name_template.format(image_num))

        
        vals_angle = np.angle(Z)
        vals_angle = vals_angle-3.14159265358979323

        image_num += 1

        Zs.append(Z)

    with gzip.open('Zs.pkl.gz', 'wb') as f:
        dill.dump(Zs, f)

    sys.exit(-1)

    cmplxs = np.empty((n, ), dtype=np.complex)
    cmplxs.real = xs
    cmplxs.imag = ys

    plt.close("all")

    plt.figure()
    plt.title("Points with complex numbers")
    plt.xlabel("real")
    plt.ylabel("imag")

    plt.plot(cmplxs.real, cmplxs.imag, "b.")
    plt.plot(cmplxs.real, cmplxs.imag, "g-")

    k = 3
    ts = np.arange(0, k)
    asz = get_factors(ts, cmplxs[0:+k])

    ts_many = np.arange(0, ts[-1], ts[-1]/100)
    ts_many = np.hstack((ts_many, (ts[-1], )))
    cmplxs_inter = get_ys(ts_many, asz)

    plt.plot(cmplxs_inter.real, cmplxs_inter.imag, "r-")


    asz_x = get_factors(ts, xs[0:0+k])
    asz_y = get_factors(ts, ys[0:0+k])

    xs1 = get_ys(ts_many, asz_x)
    ys1 = get_ys(ts_many, asz_y)

    plt.plot(xs1, ys1, "g-")

    plt.show()

    print("np.sum(cmplxs_inter.real-xs1): {}".format(np.sum(cmplxs_inter.real-xs1)))
    print("np.sum(cmplxs_inter.imag-ys1): {}".format(np.sum(cmplxs_inter.imag-ys1)))
