#! /usr/bin/python3.6

import colorsys
import datetime
import dill
import os
import string
import sys
import traceback

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont


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
    # all_symbols_64 = np.array(list(string.ascii_letters+string.digits+"-_"))
    # all_symbols_16 = np.array(list("0123456789ABCDEF"))

    # def get_random_string(n):
    #     # l = np.random.randint(0, 64, (n, ))
    #     l = np.random.randint(0, 16, (n, ))
    #     return "".join(all_symbols_16[l])

    path_image = "images/"

    if not os.path.exists(path_image):
        os.makedirs(path_image)

    # func_str = "z**2-4*z+0.1*z*complex(z.imag, z.real*3)"
    # func_str = "complex(np.sin(z.imag), z.real)*z"

    func_str = "z**3"
    # func_str = "z**3-2*z**2+3*z+complex(2, -3)"
    # func_str = "z**3-2*z**2+3*z+complex(2, -3)"

    # func_str = "z+np.sin(z.real+z.imag)+complex(z.imag*np.sin(0.1*z.real), z*np.sin(z.imag+1))"
    # func_str = "2*z**2+4*z*complex(np.sin(z.real), np.cos(z.imag)**2)"
    # func_str = "z**3-2*z**2+4*z"
    # func_str = "complex(np.sin(z.imag), z.real)*z+z*complex(z.real+z.imag*0.5, 2*np.cos(z.real))"
    func_str_complete = "lambda z: "+func_str
    def get_f(modulo, func_str_complete):
        f_str = func_str_complete
        f = eval(f_str)
        def f_temp(z):
            n_z_orig = f(z)

            length = np.abs(n_z_orig)
            scale = (length % modulo) / length
            return n_z_orig, scale
        return f_temp

    modulo = 1

    f = get_f(modulo, func_str_complete)

    plt.close("all")

    n1 = 800
    scale = 30
    x_offset = 15
    # x_offset = scale/2
    scale_y = 1.
    y_offset = scale/2*scale_y
    n1_y = int(n1*scale_y)

    delta = 0.0001

    xs1 = np.arange(0, n1+1)/n1*scale-x_offset+delta
    ys1 = np.arange(0, n1_y+1)/n1_y*scale*scale_y-y_offset+delta

    arr_complex = np.empty((ys1.shape[0], xs1.shape[0]), dtype=complex)
    arr_complex_mod = np.empty((ys1.shape[0], xs1.shape[0]), dtype=complex)

    # TODO: create the other magtrices too!

    pix = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    pix_hvs = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)
    pix_mod = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    pix_hvs_mod = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)

    pix_angle = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    pix_hvs_angle = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)
    pix_abs = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    pix_hvs_abs = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)
    pix_x = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    pix_hvs_x = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)
    pix_y = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    pix_hvs_y = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)

    print("xs1: {}".format(xs1))
    print("ys1: {}".format(ys1))

    for yi, y in enumerate(ys1):
        print("yi: {}".format(yi))
        for xi, x in enumerate(xs1):
            z, scale_z = f(complex(x, y))
            # z, scale_z = f(complex(x, y), modulo)
            arr_complex[yi, xi] = z
            z_scale = z*scale_z
            arr_complex_mod[yi, xi] = z_scale
            
            zx, zy = z.real, z.imag
            zx_s, zy_s = z_scale.real, z_scale.imag
            if zx == 0. and zy == 0.:
                continue

            alpha = calculate_angle_x_y(zx, zy)

            h = alpha/(np.pi*2)
            h = 1. if h > 1. else h

            a = np.sqrt(zx**2+zy**2)
            a_s = np.sqrt(zx_s**2+zy_s**2)

            pix_hvs[yi, xi] = (h, a, 1.)
            pix_hvs_mod[yi, xi] = (h, a_s, 1.)
            
            pix_hvs_angle[yi, xi] = (h, 1., 1.)
            pix_hvs_abs[yi, xi] = (a, 1., 1.)
            pix_hvs_x[yi, xi] = (z.real, 1., 1.)
            pix_hvs_y[yi, xi] = (z.imag, 1., 1.)

            # pix_hvs_angle[yi, xi] = (h, 1., 1.)
            # pix_hvs_abs[yi, xi] = (a, 1., 1.)
            # pix_hvs_x[yi, xi] = (z.real, 1., 1.)
            # pix_hvs_y[yi, xi] = (z.imag, 1., 1.)
    
    # pix_hvs[:, :, 1] /= np.max(pix_hvs[:, :, 1])

    def normalize_v_in_hvs(pix, pix_hvs, col=1):
        vals = pix_hvs[:, :, col].reshape((-1, ))

        combo = np.empty((0, vals.shape[0]), dtype=np.object)
        combo = np.vstack((combo, np.arange(0, vals.shape[0])))
        combo = np.vstack((combo, vals)).T

        combo = combo[combo[:, 1].argsort()]
        combo[:, 1] = combo[:, 1]-combo[0, 1]

        combo[:, 1] = (lambda x: x/x[-1])(combo[:, 1]**(1/4))/2+0.5
        vals[combo[:, 0].astype(np.int)] = combo[:, 1]

        print("combo.shape: {}".format(combo.shape))
        print("combo.dtype: {}".format(combo.dtype))

        vals_sort = np.sort(vals)
        vals_1 = vals_sort
        vals_2 = vals_1/vals_1[-1]
        x_vals = np.arange(0, vals.shape[0])

        for y in range(0, pix.shape[0]):
            print("yi2: {}".format(y))
            for x in range(0, pix.shape[1]):
                args = pix_hvs[y, x]
                c = colorsys.hsv_to_rgb(*args)
                c_uint8 = (np.array(c)*255.49).astype(np.uint8)
                pix[y, x] = c_uint8

    normalize_v_in_hvs(pix, pix_hvs, col=1)
    normalize_v_in_hvs(pix_mod, pix_hvs_mod, col=1)
    
    using_col = 0
    normalize_v_in_hvs(pix_angle, pix_hvs_angle, col=using_col)
    normalize_v_in_hvs(pix_abs, pix_hvs_abs, col=using_col)
    normalize_v_in_hvs(pix_x, pix_hvs_x, col=using_col)
    normalize_v_in_hvs(pix_y, pix_hvs_y, col=using_col)
    
    # print("np.min(pix_hvs[:, :, 0]): {}".format(np.min(pix_hvs[:, :, 0])))
    # print("np.max(pix_hvs[:, :, 0]): {}".format(np.max(pix_hvs[:, :, 0])))
    
    # print("\nnp.min(pix_hvs[:, :, 1]): {}".format(np.min(pix_hvs[:, :, 1])))
    # print("np.max(pix_hvs[:, :, 1]): {}".format(np.max(pix_hvs[:, :, 1])))
    
    # print("\nnp.min(pix_hvs[:, :, 2]): {}".format(np.min(pix_hvs[:, :, 2])))
    # print("np.max(pix_hvs[:, :, 2]): {}".format(np.max(pix_hvs[:, :, 2])))

    pix2 = pix.copy()[:, :300]
    pix2[:] = 0

    img2 = Image.fromarray(pix2)
    
    # get a font
    fnt = ImageFont.truetype('monofonto.ttf', 16)
    d = ImageDraw.Draw(img2)
    d.text((8, 8), "Used function:", font=fnt, fill=(255, 255, 255))
    func_str_split = [func_str_complete[30*i:30*(i+1)] for i in range(0, len(func_str_complete)//30+1)]

    for i, func_str_part in enumerate(func_str_split, 1):
        d.text((8, 8+24*i), func_str_part, font=fnt, fill=(255, 255, 255))
    
    font_y_next = 8+24*(i+2)
    d.text((8, font_y_next), "x_min: {:3.02f}, x_max: {:3.02f}".format(-x_offset, scale-x_offset), font=fnt, fill=(255, 255, 255))        
    
    font_y_next += 24
    d.text((8, font_y_next), "y_min: {:3.02f}, y_max: {:3.02f}".format(-y_offset, scale-y_offset), font=fnt, fill=(255, 255, 255))        
    
    pix2 = np.array(img2).copy()

    font_y_next += 24

    img2_mod = img2.copy()
    d = ImageDraw.Draw(img2_mod)
    d.text((8, font_y_next), "modulo: {:3.02f}".format(modulo), font=fnt, fill=(255, 255, 255))        
    pix2_mod = np.array(img2_mod)

    img2_angle = img2.copy()
    d = ImageDraw.Draw(img2_angle)
    d.text((8, font_y_next), "only with angle", font=fnt, fill=(255, 255, 255))        
    pix2_angle = np.array(img2_angle)

    img2_abs = img2.copy()
    d = ImageDraw.Draw(img2_abs)
    d.text((8, font_y_next), "only with abs", font=fnt, fill=(255, 255, 255))        
    pix2_abs = np.array(img2_abs)

    img2_x = img2.copy()
    d = ImageDraw.Draw(img2_x)
    d.text((8, font_y_next), "only with x", font=fnt, fill=(255, 255, 255))        
    pix2_x = np.array(img2_x)

    img2_y = img2.copy()
    d = ImageDraw.Draw(img2_y)
    d.text((8, font_y_next), "only with y", font=fnt, fill=(255, 255, 255))        
    pix2_y = np.array(img2_y)

    find_x = 0.
    find_y = 0.
    rows, cols = pix.shape[:2]
    line_col = np.argmin((xs1-find_x)**2)
    line_row = np.argmin((ys1-find_y)**2)

    if not(line_row < 0 or line_row >= rows):
        for y in range(0, rows):
            pix[y, line_col] = pix[y, line_col]^(0xFF, )*3
            pix_mod[y, line_col] = pix_mod[y, line_col]^(0xFF, )*3

            pix_angle[y, line_col] = pix_angle[y, line_col]^(0xFF, )*3
            pix_abs[y, line_col] = pix_abs[y, line_col]^(0xFF, )*3
            pix_x[y, line_col] = pix_x[y, line_col]^(0xFF, )*3
            pix_y[y, line_col] = pix_y[y, line_col]^(0xFF, )*3

    if not(line_col < 0 or line_col >= cols):
        for x in range(0, cols):
            pix[line_row, x] = pix[line_row, x]^(0xFF, )*3
            pix_mod[line_row, x] = pix_mod[line_row, x]^(0xFF, )*3

            pix_angle[line_row, x] = pix_angle[line_row, x]^(0xFF, )*3
            pix_abs[line_row, x] = pix_abs[line_row, x]^(0xFF, )*3
            pix_x[line_row, x] = pix_x[line_row, x]^(0xFF, )*3
            pix_y[line_row, x] = pix_y[line_row, x]^(0xFF, )*3

    pix = np.hstack((pix2, pix))
    pix_mod = np.hstack((pix2_mod, pix_mod))
    pix_angle = np.hstack((pix2_mod, pix_angle))
    pix_abs = np.hstack((pix2_mod, pix_abs))
    pix_x = np.hstack((pix2_mod, pix_x))
    pix_y = np.hstack((pix2_mod, pix_y))

    img = Image.fromarray(pix)
    img_mod = Image.fromarray(pix_mod)
    img_angle = Image.fromarray(pix_angle)
    img_abs = Image.fromarray(pix_abs)
    img_x = Image.fromarray(pix_x)
    img_y = Image.fromarray(pix_y)

    file_name = "{:%Y_%m_%d_%H_%M_%S_%f}.png".format(datetime.datetime.now())
    img.save(path_image+file_name)

    file_name = "{:%Y_%m_%d_%H_%M_%S_%f}.png".format(datetime.datetime.now())
    img_mod.save(path_image+file_name)


    file_name = "{:%Y_%m_%d_%H_%M_%S_%f}.png".format(datetime.datetime.now())
    img_angle.save(path_image+file_name)

    file_name = "{:%Y_%m_%d_%H_%M_%S_%f}.png".format(datetime.datetime.now())
    img_abs.save(path_image+file_name)

    file_name = "{:%Y_%m_%d_%H_%M_%S_%f}.png".format(datetime.datetime.now())
    img_x.save(path_image+file_name)

    file_name = "{:%Y_%m_%d_%H_%M_%S_%f}.png".format(datetime.datetime.now())
    img_y.save(path_image+file_name)
    
    # img.show()
    # img_mod.show()

    # TODO: Need to save this in a data format (e.g. pkl.gz with a dotmap dict)
    # TODO: saving with all important parameters, like n1, scale, x_offset, ...
    # TODO: also save the used function, also if modulo is used or not ;-)
