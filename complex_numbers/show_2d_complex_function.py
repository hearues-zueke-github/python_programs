#! /usr/bin/python2.7

import colorsys
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image


def calculate_angle_x_y(x, y):
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
    path_image = "images/"
    if not os.path.exists(path_image):
        os.makedirs(path_image)

    # def f(z):
    def f(z, m):
        # n_z_orig = np.sin(z*2)+np.cos(z.conjugate())
        # n_z_orig = complex(np.sin(z.real), np.cos(z.imag))*complex(np.sin(z.imag)*np.cos(z.imag), np.cos(z.real)*np.sin(z.imag))
        
        n_z_orig = complex(np.sin(z.real), np.cos(z.imag)*2)*z

        # n_z_orig = 3*z**2
        # n_z_orig = (z+complex(-1, -1))**2+(z+complex(+1, 0))**2
        # n_z_orig = np.sin(z.imag)*np.cos(-z.real)+z**2-3*z
        # n_z_orig = np.sin(z)*np.cos(complex(-z.imag, -z.real))
        # n_z_orig = np.sin(z*2)+np.cos(complex(z.imag, z.real))*z
        # return n_z_orig

        # TODO: scale the z to np.abs(z) % m, not real % m and imag % m
        length = np.abs(n_z_orig)
        scale = (length % m) / length
        return n_z_orig * scale

        # real = n_z_orig.real
        # imag = n_z_orig.imag
        # n_z = complex(np.sign(real)*(np.abs(real) % m),
        #               np.sign(imag)*(np.abs(imag) % m))
        # return n_z

    m = 1

    plt.close("all")

    n1 = 1200
    scale = 21
    x_offset = scale/2
    scale_y = 1.
    y_offset = scale/2*scale_y
    n1_y = int(n1*scale_y)

    delta = 0.0001

    xs1 = np.arange(0, n1+1)/n1*scale-x_offset+delta
    ys1 = np.arange(0, n1_y+1)/n1_y*scale*scale_y-y_offset+delta

    arr_complex = np.empty((ys1.shape[0], xs1.shape[0]), dtype=complex)
    pix = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    pix_hvs = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)
    print("xs1: {}".format(xs1))
    print("ys1: {}".format(ys1))

    for yi, y in enumerate(ys1):
        print("yi: {}".format(yi))
        for xi, x in enumerate(xs1):
            z = f(complex(x, y), m)
            arr_complex[yi, xi] = z
            
            zx, zy = z.real, z.imag
            if zx == 0. and zy == 0.:
                continue

            alpha = calculate_angle_x_y(zx, zy)

            h = alpha/(np.pi*2)
            h = 1. if h > 1. else h

            pix_hvs[yi, xi] = (h, np.sqrt(zx**2+zy**2), 1.)
    
    # pix_hvs[:, :, 1] /= np.max(pix_hvs[:, :, 1])

    vals = pix_hvs[:, :, 1].reshape((-1, ))

    combo = np.empty((0, vals.shape[0]), dtype=np.object)
    combo = np.vstack((combo, np.arange(0, vals.shape[0])))
    combo = np.vstack((combo, vals)).T

    combo = combo[combo[:, 1].argsort()]

    combo[:, 1] = (lambda x: x/x[-1])(combo[:, 1]**(1/4))/2+0.5
    vals[combo[:, 0].astype(np.int)] = combo[:, 1]

    print("combo.shape: {}".format(combo.shape))
    print("combo.dtype: {}".format(combo.dtype))

    vals_sort = np.sort(vals)
    vals_1 = vals_sort**(1/4)
    vals_2 = vals_1/vals_1[-1]
    x_vals = np.arange(0, vals.shape[0])

    for y in range(0, pix.shape[0]):
        print("yi2: {}".format(y))
        for x in range(0, pix.shape[1]):
            args = pix_hvs[y, x]
            c = colorsys.hsv_to_rgb(*args)
            c_uint8 = (np.array(c)*255.49).astype(np.uint8)
            pix[y, x] = c_uint8
    
    print("np.min(pix_hvs[:, :, 0]): {}".format(np.min(pix_hvs[:, :, 0])))
    print("np.max(pix_hvs[:, :, 0]): {}".format(np.max(pix_hvs[:, :, 0])))
    
    print("\nnp.min(pix_hvs[:, :, 1]): {}".format(np.min(pix_hvs[:, :, 1])))
    print("np.max(pix_hvs[:, :, 1]): {}".format(np.max(pix_hvs[:, :, 1])))
    
    print("\nnp.min(pix_hvs[:, :, 2]): {}".format(np.min(pix_hvs[:, :, 2])))
    print("np.max(pix_hvs[:, :, 2]): {}".format(np.max(pix_hvs[:, :, 2])))

    img = Image.fromarray(pix)
    img.save(path_image+"complex_scale_{}_x_off_{}_y_off_{}.png"
        .format(
            "{:02.02f}".format(scale).replace(".", "_"),
            "{:02.02f}".format(x_offset).replace(".", "_"),
            "{:02.02f}".format(y_offset).replace(".", "_")
        ))

    img.show()

    # TODO: Need to save this in a data format (e.g. pkl.gz with a dotmap dict)
    # TODO: saving with all important parameters, like n1, scale, x_offset, ...
    # TODO: also save the used function, also if modulo is used or not ;-)
