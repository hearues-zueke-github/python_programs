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
    all_symbols_16 = np.array(list("0123456789ABCDEF"))

    def get_random_string(n):
        # l = np.random.randint(0, 64, (n, ))
        l = np.random.randint(0, 16, (n, ))
        return "".join(all_symbols_16[l])

    path_image = "images/random_names/"

    # rand_str = get_random_string(16)
    # print("rand_str: {}".format(rand_str))
    # path_image += rand_str+"/"

    if not os.path.exists(path_image):
        os.makedirs(path_image)

    func_str = "z**2-4*z+0.1*z*complex(z.imag, z.real*3)"
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

    n1 = 500
    scale = 40
    x_offset = scale/2
    scale_y = 1.
    y_offset = scale/2*scale_y
    n1_y = int(n1*scale_y)

    delta = 0.0001

    xs1 = np.arange(0, n1+1)/n1*scale-x_offset+delta
    ys1 = np.arange(0, n1_y+1)/n1_y*scale*scale_y-y_offset+delta

    arr_complex = np.empty((ys1.shape[0], xs1.shape[0]), dtype=complex)
    arr_complex_mod = np.empty((ys1.shape[0], xs1.shape[0]), dtype=complex)

    pix = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.uint8)
    pix_hvs = np.zeros((ys1.shape[0], xs1.shape[0], 3), dtype=np.float)

    print("xs1: {}".format(xs1))
    print("ys1: {}".format(ys1))

    for yi, y in enumerate(ys1):
        print("yi: {}".format(yi))
        for xi, x in enumerate(xs1):
            z, scale_z = f(complex(x, y))
            # z, scale_z = f(complex(x, y), modulo)
            arr_complex[yi, xi] = z
            arr_complex_mod[yi, xi] = z*scale_z
            
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
    vals_1 = vals_sort#**(1/4)
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

    pix2 = pix.copy()[:, :300]
    pix2[:] = 0

    # img = Image.fromarray(pix)
    img2 = Image.fromarray(pix2)
    
    try:
        # get a font
        fnt = ImageFont.truetype('monofonto.ttf', 16)
        d = ImageDraw.Draw(img2)
        d.text((8, 8), "Used function:", font=fnt, fill=(255, 255, 255))
        func_str_split = [func_str_complete[30*i:30*(i+1)] for i in range(0, len(func_str_complete)//30+1)]

        for i, func_str_part in enumerate(func_str_split, 1):
            d.text((8, 8+24*i), func_str_part, font=fnt, fill=(255, 255, 255))
        font_y_next = 8+24*(i+2)
        d.text((8, font_y_next), "n1: {:4}".format(n1), font=fnt, fill=(255, 255, 255))

        font_y_next += 24
        d.text((8, font_y_next), "scale: {:3.02f}".format(scale), font=fnt, fill=(255, 255, 255))
        
        font_y_next += 24
        d.text((8, font_y_next), "x_offset: {:3.02f}".format(x_offset), font=fnt, fill=(255, 255, 255))
        
        font_y_next += 24
        d.text((8, font_y_next), "scale_y: {:3.02f}".format(scale_y), font=fnt, fill=(255, 255, 255))        
        
        font_y_next += 24
        d.text((8, font_y_next), "y_offset: {:3.02f}".format(y_offset), font=fnt, fill=(255, 255, 255))        

        font_y_next += 24
        d.text((8, font_y_next), "n1_y: {:3.02f}".format(n1_y), font=fnt, fill=(255, 255, 255))        
        
        font_y_next += 24
        d.text((8, font_y_next), "delta: {:3.02f}".format(delta), font=fnt, fill=(255, 255, 255))        

        font_y_next += 24
        d.text((8, font_y_next), "modulo: {:3.02f}".format(modulo), font=fnt, fill=(255, 255, 255))        
        
        font_y_next += 24*2
        d.text((8, font_y_next), "x_min: {:3.02f}, x_max: {:3.02f}".format(-x_offset, scale-x_offset), font=fnt, fill=(255, 255, 255))        
        
        font_y_next += 24
        d.text((8, font_y_next), "y_min: {:3.02f}, y_max: {:3.02f}".format(-y_offset, scale-y_offset), font=fnt, fill=(255, 255, 255))        
    except:
        print("Something is wrong!!!")

        traceback.print_exc()

    pix2 = np.array(img2)

    pix = np.hstack((pix2, pix))

    # # find the pixels of the font
    # font_pix_pos = np.where(np.sum(pix2!=(0, 0, 0), axis=2)>0)

    # for y, x in zip(*font_pix_pos):
    #     pix[y, x] = (((pix[y, x]^0xFF).astype(np.float)+(pix2[y, x]).astype(np.float))/2).astype(np.uint8)

    # # and then invert the pixel at the position!

    img = Image.fromarray(pix)

    file_name = "{:%Y_%m_%d}_{}.png".format(datetime.datetime.now(), get_random_string(16))
    img.save(path_image+file_name)
    # img.save(path_image+"complex_scale_{}_x_off_{}_y_off_{}.png"
    #     .format(
    #         "{:02.02f}".format(scale).replace(".", "_"),
    #         "{:02.02f}".format(x_offset).replace(".", "_"),
    #         "{:02.02f}".format(y_offset).replace(".", "_")
    #     ))

    # img2.show()
    img.show()

    # TODO: Need to save this in a data format (e.g. pkl.gz with a dotmap dict)
    # TODO: saving with all important parameters, like n1, scale, x_offset, ...
    # TODO: also save the used function, also if modulo is used or not ;-)
