#! /usr/bin/python2.7

import colorsys
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image


def calcualte_angle_x_y(x, y):
    if x >= 0:
        if y >= 0:
            if x >= y:
                return np.arctan(y/x)
            else:
                return np.pi/2-np.arctan(x/y)
        else:
            y = -y
            if x >= y:
                return np.pi+np.pi-np.arctan(y/x)
            else:
                return np.pi+np.arctan(x/y)+np.pi/2
    else:
        x = -x
        if y >= 0:
            if x >= y:
                return np.pi-np.arctan(y/x)
            else:
                return np.arctan(x/y)+np.pi/2
        else:
            y = -y
            if x >= y:
                return np.pi+np.arctan(y/x)
            else:
                return np.pi+np.pi/2-np.arctan(x/y)

    return 0.


if __name__ == "__main__":
    # print("Hello World!")

    path_image = "image/"
    if not os.path.exists(path_image):
        os.makedirs(path_image)

    f = lambda z: complex(z.real, 4)*z*complex(3, z.imag)-z**3+0.4*z**4
    # f = lambda z: 1.9*z**2+3*z+complex(4, -1)+0.5*z**5-0.15*z**4+z*complex(z.real, -2)-complex(-2, 4)*z*complex(3, z.imag)
    # f = lambda z: 1.9*z**2+3*z+complex(4, -1)+0.5*z**5-0.15*z**4

    z = complex(3, 8)
    print("z: {}".format(z))
    z1 = f(z)
    print("z1: {}".format(z1))

    n = 100
    ts = np.arange(0, n+3)/n*np.pi*2
    print("ts: {}".format(ts))

    xs = np.cos(ts)
    ys = np.sin(ts)

    plt.close("all")
    # plt.figure()

    # plt.title("Simple Circle")
    # plt.plot(xs, ys, "b.")
    # plt.grid(True)
    # plt.axes().set_aspect('equal', 'datalim')

    # plt.figure()

    # plt.title("Angles")
    # plt.plot(ts, [calcualte_angle_x_y(x, y) for x, y in zip(xs, ys)], "g.")
    # plt.grid(True)
    # plt.axes().set_aspect('equal', 'datalim')


    n1 = 500
    scale = 20
    x_offset = 10.
    y_offset = 10.

    xs1 = np.arange(0, n1+1)/n1*scale-x_offset
    ys1 = np.arange(0, n1+1)/n1*scale-y_offset

    arr_complex = np.empty((ys1.shape[0], xs1.shape[0]), dtype=complex)
    pix = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    pix_hvs = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)
    print("xs1: {}".format(xs1))
    print("ys1: {}".format(ys1))

    # plt.figure()

    # plt.title("Output of complex numbers")
    # for x, y in zip(xs, ys):
    for yi, y in enumerate(ys1):
        print("yi: {}".format(yi))
        for xi, x in enumerate(xs1):
            z = f(complex(x, y))
            arr_complex[yi, xi] = z
            zx, zy = z.real, z.imag
            if zx == 0. and zy == 0.:
                continue
            alpha = calcualte_angle_x_y(zx, zy)
            # alpha = calcualte_angle_x_y(x, y)
            # print("y: {}, x: {}, alpha: {}".format(y, x, alpha))
            h = alpha/(np.pi*2)
            h = 1. if h > 1. else h
            # alpha = calcualte_angle_x_y(zx, zy)
            args = (h, np.sqrt(zx**2+zy**2), 1.)
            pix_hvs[yi, xi] = args
            # print("c: {}".format(c))
    
    # pix_hvs[:, :, 1] /= np.max(pix_hvs[:, :, 1])

    vals = pix_hvs[:, :, 1].reshape((-1, ))

    combo = np.empty((0, vals.shape[0]), dtype=np.object)
    combo = np.vstack((combo, np.arange(0, vals.shape[0])))
    combo = np.vstack((combo, vals)).T

    combo = combo[combo[:, 1].argsort()]

    combo[:, 1] = (lambda x: x/x[-1])(combo[:, 1]**(1/4))/2+0.5
    # pix_hvs[]
    vals[combo[:, 0].astype(np.int)] = combo[:, 1]

    # vals_xy = ( np.vstack((np.arange(0, vals.shape[0]).astype(object), vals.astype(object))).T
        # .reshape((-1, ))
        # .view("o,o") )
    # vals_xy_sorted = np.sort(vals_xy, order=["f1", "f0"])

    # print("vals_xy.shape: {}".format(vals_xy.shape))
    # print("vals_xy.dtype: {}".format(vals_xy.dtype))

    print("combo.shape: {}".format(combo.shape))
    print("combo.dtype: {}".format(combo.dtype))

    # sys.exit(-1)

    vals_sort = np.sort(vals)
    vals_1 = vals_sort**(1/4)
    vals_2 = vals_1/vals_1[-1]
    x_vals = np.arange(0, vals.shape[0])

    plt.figure()

    plt.title("values:")
    # plt.plot(x_vals, vals_sort, "b.")
    plt.plot(x_vals, vals_2, "g.")

    plt.show(block=False)

    for y in range(0, pix.shape[0]):
        print("yi2: {}".format(y))
        for x in range(0, pix.shape[1]):
            args = pix_hvs[y, x]
            # print("args: {}".format(args))
            c = colorsys.hsv_to_rgb(*args)
            # print("c: {}".format(c))
            c_uint8 = (np.array(c)*255.49).astype(np.uint8)
            # plt.plot(x, y, ".", color=c) #color=(0.4, 0.3, alpha/(np.pi*2)))
            pix[y, x] = c_uint8
            # plt.plot(z.real, z.imag, ".", color=c) #color=(0.4, 0.3, alpha/(np.pi*2)))
    
    print("np.min(pix_hvs[:, :, 0]): {}".format(np.min(pix_hvs[:, :, 0])))
    print("np.max(pix_hvs[:, :, 0]): {}".format(np.max(pix_hvs[:, :, 0])))
    
    print("\nnp.min(pix_hvs[:, :, 1]): {}".format(np.min(pix_hvs[:, :, 1])))
    print("np.max(pix_hvs[:, :, 1]): {}".format(np.max(pix_hvs[:, :, 1])))
    
    print("\nnp.min(pix_hvs[:, :, 2]): {}".format(np.min(pix_hvs[:, :, 2])))
    print("np.max(pix_hvs[:, :, 2]): {}".format(np.max(pix_hvs[:, :, 2])))

    # xsn, ysn = list(zip(*[(lambda z: (z.real, z.imag))(f(complex(x, y))) for x, y in zip(xs, ys)]))
    # plt.plot(xsn, ysn, ".", color=(0.4, 0.3, 0.1))
   
    # plt.grid(True)
    # plt.axes().set_aspect('equal', 'datalim')


    # plt.figure()

    # plt.title("Angles")
    # plt.plot(ts, [calcualte_angle_x_y(x, y) for x, y in zip(xsn, ysn)], "k.")
    # plt.grid(True)
    # plt.axes().set_aspect('equal', 'datalim')

    # plt.show()

    img = Image.fromarray(pix)
    img.save(path_image+"complex_scale_{}_x_off_{}_y_off_{}.png"
        .format(
            "{:02.02f}".format(scale).replace(".", "_"),
            "{:02.02f}".format(x_offset).replace(".", "_"),
            "{:02.02f}".format(y_offset).replace(".", "_")
        ))
    img.show()
