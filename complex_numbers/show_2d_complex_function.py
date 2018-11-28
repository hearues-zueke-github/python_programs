#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

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

# TODO: either fix this a bit or use np.angle!
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
    all_symbols_16 = np.array(list("0123456789ABCDEF"))

    def get_random_string(n):
        # l = np.random.randint(0, 64, (n, ))
        l = np.random.randint(0, 16, (n, ))
        return "".join(all_symbols_16[l])

    # func_str = "z**2-4*z+0.1*z*complex(z.imag, z.real*3)"
    # func_str = "complex(np.sin(z.imag), z.real)*z"

    # func_str = "2*z**3-complex(z.imag*np.sin(0.1*z.real), 3+z.imag)*complex(3*z.real, z.real-z.imag)"
    # func_str = "z**3-2*z**2+3*z+complex(2, -3)"
    
    # TODO: try to make a generic z-function!
    with open("z_func.txt", "r") as fin:
        func_str = fin.read()
    print("func_str: {}".format(func_str))
    
    # sys.exit(-1)
    # func_str = "z/( complex(np.abs(z.real), np.abs(z.imag))+complex(1, 1) )+z**3*0.1+complex(2, 1)"
    
    # func_str = "z**1-complex(z*np.sin(x), np.cos(z.real+z.imag*2))+np.cos(z.real)-complex(np.sin(z.imag+z.real), 3*z)*complex(z.imag, z.imag+z.real*0.1)"
    # func_str = "z*complex(np.sqrt(np.abs(z.imag))*1, np.cos(z.real))"
    # func_str = "z*complex(np.sin(z.imag), np.cos(z.real))"
    
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
            
            # Is needed for a continous modulo calculation!
            length_mod = length % modulo
            scale = (length_mod if int(length/modulo)%2==0 else modulo-length_mod) / length
            
            # Normal modulo calc
            # scale = (length % modulo) / length
            
            return n_z_orig, scale
        return f_temp

    modulo = 1.

    f = get_f(modulo, func_str_complete)

    plt.close("all")

    n1 = 300
    scale = 16
    scale_y = 1.
    
    x_offset = 8
    y_offset = 8
    n1_y = int(n1*scale_y)

    delta = 0.0001

    xs1 = np.arange(0, n1+1)/n1*scale-x_offset+delta
    ys1 = np.arange(0, n1_y+1)/n1_y*scale*scale_y-y_offset+delta

    # arr_complex = np.empty((ys1.shape[0], xs1.shape[0]), dtype=complex)
    # arr_complex_mod = np.empty((ys1.shape[0], xs1.shape[0]), dtype=complex)

    # TODO: create the other magtrices too!

    # pix = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    # pix_hvs = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)
    # pix_mod = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    # pix_hvs_mod = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)

    # pix_angle = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    # pix_hvs_angle = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)
    # pix_abs = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    # pix_hvs_abs = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)
    # pix_x = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    # pix_hvs_x = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)
    # pix_y = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    # pix_hvs_y = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)

    # print("xs1: {}".format(xs1))
    # print("ys1: {}".format(ys1))

    ys1_2d = np.zeros((ys1.shape[0], xs1.shape[0]))
    xs1_2d = ys1_2d.copy()

    xs1_2d[:] = xs1 
    ys1_2d[:] = ys1.reshape((-1, 1))
    arr_complex = np.vectorize(complex)(xs1_2d, ys1_2d)


    print("Calculate arr_f and arr_scales")
    arr_f, arr_scales = np.vectorize(f)(arr_complex)


    print("Calculate arr")
    arr_angle = np.angle(arr_f)
    idx = arr_angle < 0
    arr_angle[idx] = arr_angle[idx]+np.pi

    arr_angle_norm = arr_angle/np.max(arr_angle)


    print("Calculate abs")
    arr_abs = np.abs(arr_f)
    arr_abs_mod = arr_abs*arr_scales

    arr_abs_norm = arr_abs/np.max(arr_abs)
    arr_abs_mod_norm = arr_abs_mod/np.max(arr_abs_mod)


    print("Calculate x and y")
    f_norm_axis = lambda v: (lambda v2: v/np.max(v))(v-np.min(v))
    arr_x = arr_f.real
    arr_y = arr_f.imag
    arr_x_mod = arr_x*arr_scales
    arr_y_mod = arr_y*arr_scales

    arr_x_norm = f_norm_axis(arr_x)
    arr_y_norm = f_norm_axis(arr_y)
    arr_x_mod_norm = f_norm_axis(arr_x_mod)
    arr_y_mod_norm = f_norm_axis(arr_y_mod)


    print("Getting all data in pix_float array")
    pix_float_hsv_f = np.ones((arr_complex.shape+(3, )))
    pix_float_hsv_f_mod = np.ones((arr_complex.shape+(3, )))
    pix_float_hsv_angle = np.ones((arr_complex.shape+(3, )))
    pix_float_hsv_abs = np.ones((arr_complex.shape+(3, )))
    pix_float_hsv_abs_mod = np.ones((arr_complex.shape+(3, )))
    pix_float_hsv_x = np.ones((arr_complex.shape+(3, )))
    pix_float_hsv_x_mod = np.ones((arr_complex.shape+(3, )))
    pix_float_hsv_y = np.ones((arr_complex.shape+(3, )))
    pix_float_hsv_y_mod = np.ones((arr_complex.shape+(3, )))

    pix_float_hsv_f[:, :, 0] = arr_angle_norm
    pix_float_hsv_f[:, :, 1] = arr_abs_norm
    
    pix_float_hsv_f_mod[:, :, 0] = arr_angle_norm
    pix_float_hsv_f_mod[:, :, 1] = arr_abs_mod_norm

    pix_float_hsv_angle[:, :, 0] = arr_angle_norm

    pix_float_hsv_abs[:, :, 0] = arr_abs_norm
    pix_float_hsv_abs_mod[:, :, 0] = arr_abs_mod_norm

    pix_float_hsv_x[:, :, 0] = arr_x_norm
    pix_float_hsv_x_mod[:, :, 0] = arr_x_mod_norm

    pix_float_hsv_y[:, :, 0] = arr_y_norm
    pix_float_hsv_y_mod[:, :, 0] = arr_y_mod_norm

    # print("pix_float_hsv_f.shape: {}".format(pix_float_hsv_f.shape))
    # print("pix_float_hsv_mod_f.shape: {}".format(pix_float_hsv_mod_f.shape))


    hsv_to_rgb_vectorized = np.vectorize(colorsys.hsv_to_rgb)

    print("Convert hsv to rgb and convert to uint8")
    pix_float_rgb_f = np.dstack(hsv_to_rgb_vectorized(pix_float_hsv_f[..., 0], pix_float_hsv_f[..., 1], pix_float_hsv_f[..., 2]))
    pix_rgb_f = (pix_float_rgb_f*255.9).astype(np.uint8)

    pix_float_rgb_f_mod = np.dstack(hsv_to_rgb_vectorized(pix_float_hsv_f_mod[..., 0], pix_float_hsv_f_mod[..., 1], pix_float_hsv_f_mod[..., 2]))
    pix_rgb_f_mod = (pix_float_rgb_f_mod*255.9).astype(np.uint8)

    pix_float_rgb_angle = np.dstack(hsv_to_rgb_vectorized(pix_float_hsv_angle[..., 0], pix_float_hsv_angle[..., 1], pix_float_hsv_angle[..., 2]))
    pix_rgb_angle = (pix_float_rgb_angle*255.9).astype(np.uint8)
    
    pix_float_rgb_abs = np.dstack(hsv_to_rgb_vectorized(pix_float_hsv_abs[..., 0], pix_float_hsv_abs[..., 1], pix_float_hsv_abs[..., 2]))
    pix_rgb_abs = (pix_float_rgb_abs*255.9).astype(np.uint8)
    
    pix_float_rgb_abs_mod = np.dstack(hsv_to_rgb_vectorized(pix_float_hsv_abs_mod[..., 0], pix_float_hsv_abs_mod[..., 1], pix_float_hsv_abs_mod[..., 2]))
    pix_rgb_abs_mod = (pix_float_rgb_abs_mod*255.9).astype(np.uint8)

    pix_float_rgb_x = np.dstack(hsv_to_rgb_vectorized(pix_float_hsv_x[..., 0], pix_float_hsv_x[..., 1], pix_float_hsv_x[..., 2]))
    pix_rgb_x = (pix_float_rgb_x*255.9).astype(np.uint8)

    pix_float_rgb_x_mod = np.dstack(hsv_to_rgb_vectorized(pix_float_hsv_x_mod[..., 0], pix_float_hsv_x_mod[..., 1], pix_float_hsv_x_mod[..., 2]))
    pix_rgb_x_mod = (pix_float_rgb_x_mod*255.9).astype(np.uint8)

    pix_float_rgb_y = np.dstack(hsv_to_rgb_vectorized(pix_float_hsv_y[..., 0], pix_float_hsv_y[..., 1], pix_float_hsv_y[..., 2]))
    pix_rgb_y = (pix_float_rgb_y*255.9).astype(np.uint8)

    pix_float_rgb_y_mod = np.dstack(hsv_to_rgb_vectorized(pix_float_hsv_y_mod[..., 0], pix_float_hsv_y_mod[..., 1], pix_float_hsv_y_mod[..., 2]))
    pix_rgb_y_mod = (pix_float_rgb_y_mod*255.9).astype(np.uint8)

    sys.exit(-1)

    # print("doing other stuff!")
    # for yi, y in enumerate(ys1):
    #     if yi % 100 == 0:
    #         print("yi: {}".format(yi))
    #     for xi, x in enumerate(xs1):
    #         z, scale_z = f(complex(x, y))
    #         # z, scale_z = f(complex(x, y), modulo)
    #         arr_complex[yi, xi] = z
    #         z_scale = z*scale_z
    #         arr_complex_mod[yi, xi] = z_scale
            
    #         zx, zy = z.real, z.imag
    #         zx_s, zy_s = z_scale.real, z_scale.imag
    #         if zx == 0. and zy == 0.:
    #             continue

    #         alpha = calculate_angle_x_y(zx, zy)

    #         h = alpha/(np.pi*2)
    #         h = 1. if h > 1. else h

    #         a = np.sqrt(zx**2+zy**2)
    #         a_s = np.sqrt(zx_s**2+zy_s**2)

    #         pix_hvs[yi, xi] = (h, a, 1.)
    #         pix_hvs_mod[yi, xi] = (h, a_s, 1.)
            
    #         pix_hvs_angle[yi, xi] = (h, 1., 1.)
    #         pix_hvs_abs[yi, xi] = (a, 1., 1.)
    #         pix_hvs_x[yi, xi] = (z.real, 1., 1.)
    #         pix_hvs_y[yi, xi] = (z.imag, 1., 1.)


    # def normalize_v_in_hvs(pix, pix_hvs, col=1, move_to_zero=False, special_normalization=False):
    #     vals = pix_hvs[:, :, col].reshape((-1, ))

    #     combo = np.empty((0, vals.shape[0]), dtype=np.object)
    #     combo = np.vstack((combo, np.arange(0, vals.shape[0])))
    #     combo = np.vstack((combo, vals)).T

    #     combo = combo[combo[:, 1].argsort()]
    #     if move_to_zero:
    #         combo[:, 1] = combo[:, 1]-combo[0, 1]

    #     if special_normalization:
    #         combo[:, 1] = (lambda x: x/x[-1])(combo[:, 1])*1/3+2/3
    #         # combo[:, 1] = (lambda x: x/x[-1])(combo[:, 1]**(1/4))*2/3+1/3
    #     else:
    #         combo[:, 1] = (lambda x: x/x[-1])(combo[:, 1])

    #     vals[combo[:, 0].astype(np.int)] = combo[:, 1]

    #     # print("combo.shape: {}".format(combo.shape))
    #     # print("combo.dtype: {}".format(combo.dtype))

    #     vals_sort = np.sort(vals)
    #     vals_1 = vals_sort
    #     vals_2 = vals_1/vals_1[-1]
    #     x_vals = np.arange(0, vals.shape[0])

    #     for y in range(0, pix.shape[0]):
    #         # print("yi2: {}".format(y))
    #         for x in range(0, pix.shape[1]):
    #             args = pix_hvs[y, x]
    #             c = colorsys.hsv_to_rgb(*args)
    #             c_uint8 = (np.array(c)*255.49).astype(np.uint8)
    #             pix[y, x] = c_uint8

    # print("normalize pix_hvs")
    # normalize_v_in_hvs(pix, pix_hvs, col=1, special_normalization=True)
    # print("normalize pix_hvs_mod")
    # normalize_v_in_hvs(pix_mod, pix_hvs_mod, col=1, special_normalization=True)
    
    # using_col = 0
    # print("normalize pix_hvs_angle")
    # normalize_v_in_hvs(pix_angle, pix_hvs_angle, col=using_col)
    # print("normalize pix_hvs_abs")
    # normalize_v_in_hvs(pix_abs, pix_hvs_abs, col=using_col)
    # print("normalize pix_hvs_x")
    # normalize_v_in_hvs(pix_x, pix_hvs_x, col=using_col, move_to_zero=True)
    # print("normalize pix_hvs_y")
    # normalize_v_in_hvs(pix_y, pix_hvs_y, col=using_col, move_to_zero=True)
    
    # print("np.min(pix_hvs[:, :, 0]): {}".format(np.min(pix_hvs[:, :, 0])))
    # print("np.max(pix_hvs[:, :, 0]): {}".format(np.max(pix_hvs[:, :, 0])))
    
    # print("\nnp.min(pix_hvs[:, :, 1]): {}".format(np.min(pix_hvs[:, :, 1])))
    # print("np.max(pix_hvs[:, :, 1]): {}".format(np.max(pix_hvs[:, :, 1])))
    
    # print("\nnp.min(pix_hvs[:, :, 2]): {}".format(np.min(pix_hvs[:, :, 2])))
    # print("np.max(pix_hvs[:, :, 2]): {}".format(np.max(pix_hvs[:, :, 2])))

    pix2 = np.zeros((pix_rgb_f.shape[0], 300, 3), dtype=np.uint8)
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
    pix_angle = np.hstack((pix2_angle, pix_angle))
    pix_abs = np.hstack((pix2_abs, pix_abs))
    pix_x = np.hstack((pix2_x, pix_x))
    pix_y = np.hstack((pix2_y, pix_y))

    img = Image.fromarray(pix)
    img_mod = Image.fromarray(pix_mod)
    img_angle = Image.fromarray(pix_angle)
    img_abs = Image.fromarray(pix_abs)
    img_x = Image.fromarray(pix_x)
    img_y = Image.fromarray(pix_y)

    get_date_time_str = lambda: "{:%Y_%m_%d_%H_%M_%S_%f}".format(datetime.datetime.now())

    path_image = "images/{}/".format(get_date_time_str())
    # path_image = "images/{}/".format(get_random_string(16))
    if not os.path.exists(path_image):
        os.makedirs(path_image)

    file_name = get_date_time_str()+".png"
    img.save(path_image+file_name)

    file_name = get_date_time_str()+".png"
    img_mod.save(path_image+file_name)

    file_name = get_date_time_str()+".png"
    img_angle.save(path_image+file_name)

    file_name = get_date_time_str()+".png"
    img_abs.save(path_image+file_name)

    file_name = get_date_time_str()+".png"
    img_x.save(path_image+file_name)

    file_name = get_date_time_str()+".png"
    img_y.save(path_image+file_name)
    
    # img.show()
    # img_mod.show()

    # TODO: Need to save this in a data format (e.g. pkl.gz with a dotmap dict)
    # TODO: saving with all important parameters, like n1, scale, x_offset, ...
    # TODO: also save the used function, also if modulo is used or not ;-)
