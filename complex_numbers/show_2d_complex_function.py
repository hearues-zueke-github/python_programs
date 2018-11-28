#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import colorsys
import datetime
import dill
import gzip
import os
import shutil
import string
import sys
import traceback

import numpy as np

import matplotlib.pyplot as plt

from dotmap import DotMap
from PIL import Image, ImageDraw, ImageFont

import generate_generic_z_function

class GenerateComplexPictures(Exception):
    def __init__(self, modulo=10., n1=500, scale=16, scale_y=1., x_offset=8, y_offset=8, delta=0.0001):
        self.modulo = modulo
        self.n1 = n1
        self.scale = scale
        self.scale_y = scale_y
        
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.n1_y = int(self.n1*self.scale_y)

        self.delta = delta

        self.xs1 = np.arange(0, self.n1+1)/self.n1*self.scale-self.x_offset+self.delta
        self.ys1 = np.arange(0, self.n1_y+1)/self.n1_y*self.scale*self.scale_y-self.y_offset+self.delta

        self.ys1_2d = np.zeros((self.ys1.shape[0], self.xs1.shape[0]))
        self.xs1_2d = self.ys1_2d.copy()

        self.xs1_2d[:] = self.xs1 
        self.ys1_2d[:] = self.ys1.reshape((-1, 1))
        self.arr_complex = np.vectorize(complex)(self.xs1_2d, self.ys1_2d)


        self.all_symbols_16 = np.array(list("0123456789ABCDEF"))
        self.all_symbols_64 = np.array(list(string.ascii_letters+string.digits+"-_"))


        self.message = self._construct_message()
        super(GenerateComplexPictures, self).__init__(self.message)

        self.path_folder_images = "images/{}_{}/".format(
            self.get_date_time_str(),
            self.get_random_string_base_16(16)
        )
        if not os.path.exists(self.path_folder_images):
            os.makedirs(self.path_folder_images)

        generate_generic_z_function.main(path_folder=self.path_folder_images)

        with open(self.path_folder_images+"z_func.txt", "r") as fin:
            self.func_str = fin.read()
        print("func_str: {}".format(self.func_str))
        self.func_str_complete = "lambda z: "+self.func_str

        self.f = self.get_f()


    def get_random_string_base_16(self, n):
        l = np.random.randint(0, 16, (n, ))
        return "".join(self.all_symbols_16[l])


    def delete_image_folder(self):
        path_folder_images = self.path_folder_images

        print("Remove the whole folder '{}'!".format(path_folder_images))
        shutil.rmtree(path_folder_images)


    def get_random_string_base_64(self, n):
        l = np.random.randint(0, 64, (n, ))
        return "".join(self.all_symbols_64[l])


    def get_date_time_str(self):
        return "{:%Y_%m_%d_%H_%M_%S_%f}".format(datetime.datetime.now())


    def get_f(self):
        modulo = self.modulo
        f_str = self.func_str_complete
        f = eval(f_str)
        def f_temp(z):
            n_z_orig = f(z)

            length = np.abs(n_z_orig)
            
            # Is needed for a continous modulo calculation!
            length_mod = length % modulo
            scale = (length_mod if int(length/modulo)%2==0 else modulo-length_mod) / length
            
            return n_z_orig, scale
        return f_temp


    def do_calculations(self):
        arr_complex = self.arr_complex

        print("Calculate arr_f and arr_scales")
        arr_f, arr_scales = np.vectorize(self.f)(arr_complex)


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

        pixs1 = [pix_rgb_f,
                 pix_rgb_f_mod,
                 pix_rgb_angle,
                 pix_rgb_abs,
                 pix_rgb_abs_mod,
                 pix_rgb_x,
                 pix_rgb_x_mod,
                 pix_rgb_y,
                 pix_rgb_y_mod]


        print("Adding coordinates x and y if possible")
        find_x = 0.
        find_y = 0.
        rows, cols = pixs1[0].shape[:2]
        line_col = np.argmin((self.xs1-find_x)**2)
        line_row = np.argmin((self.ys1-find_y)**2)

        if not(line_row < 0 or line_row >= rows):
            for pix in pixs1:
                pix[:, line_col] = pix[:, line_col]^(0xFF, )*3

        if not(line_col < 0 or line_col >= cols):
            for pix in pixs1:
                pix[line_row, :] = pix[line_row, :]^(0xFF, )*3

        print("Creating the side infos")
        # this is needed for the info on the left side of the graph!
        pix2 = np.zeros((pix_rgb_f.shape[0], 300, 3), dtype=np.uint8)
        img2 = Image.fromarray(pix2)
        
        # get a font
        fnt = ImageFont.truetype('monofonto.ttf', 16)
        d = ImageDraw.Draw(img2)
        d.text((8, 8), "Used function:", font=fnt, fill=(255, 255, 255))

        func_str_split = [self.func_str_complete[30*i:30*(i+1)] for i in range(0, len(self.func_str_complete)//30+1)]
        for i, func_str_part in enumerate(func_str_split, 1):
            d.text((8, 8+24*i), func_str_part, font=fnt, fill=(255, 255, 255))
        
        font_y_next = 8+24*(i+2)
        d.text((8, font_y_next), "x_min: {:3.02f}, x_max: {:3.02f}".format(-self.x_offset, self.scale-self.x_offset), font=fnt, fill=(255, 255, 255))        
        
        font_y_next += 24
        d.text((8, font_y_next), "y_min: {:3.02f}, y_max: {:3.02f}".format(-self.y_offset, self.scale-self.y_offset), font=fnt, fill=(255, 255, 255))        
        
        pix2 = np.array(img2).copy()

        font_y_next += 24

        modulo_str = "modulo: {:3.02f}".format(self.modulo)
        text_to_write_lst = [
            ("only with f(z)", ),
            ("only with f(z)", modulo_str, ),
            ("only with angle", ),
            ("only with abs", ),
            ("only with abs", modulo_str),
            ("only with x", ),
            ("only with x", modulo_str),
            ("only with y", ),
            ("only with y", modulo_str)
        ]

        pixs2 = []
        for text_to_write in text_to_write_lst:
            img2_temp = img2.copy()
            d = ImageDraw.Draw(img2_temp)
            
            font_y_next_temp = font_y_next
            for text in text_to_write:
                d.text((8, font_y_next_temp), text, font=fnt, fill=(255, 255, 255))        
                font_y_next_temp += 24

            pixs2.append(np.array(img2_temp))

        imgs = []
        for pix2, pix1 in zip(pixs2, pixs1):
            pix3 = np.hstack((pix2, pix1))
            imgs.append(Image.fromarray(pix3))

        suffixes = ["f", "f_mod", "angle", "abs", "abs_mod", "x", "x_mod", "y", "y_mod"]
        for suffix, img in zip(suffixes, imgs):
            file_name = "_{}.png".format(suffix)
            img.save(self.path_folder_images+
                "{}_{}".format(
                    self.get_date_time_str(),
                    self.get_random_string_base_16(16)
                )+
                file_name
            )


    def save_new_z_function(self):
        path_folder_data = "data/"
        if not os.path.exists(path_folder_data):
            os.makedirs(path_folder_data)

        path_file_data = path_folder_data+"working_z_functions.pkl.gz"

        if not os.path.exists(path_file_data):
            data = DotMap()
            data.func_str_lst = [self.func_str_complete]

            with gzip.open(path_file_data, "wb") as fout:
                dill.dump(data, fout)
        else:
            with gzip.open(path_file_data, "rb") as fin:
                data = dill.load(fin)

            data.func_str_lst.append(self.func_str_complete)
            
            with gzip.open(path_file_data, "wb") as fout:
                dill.dump(data, fout)

        print("data.func_str_lst:\n{}".format(data.func_str_lst))

        lst = data.func_str_lst
        print("Amount of found functions: {}".format(len(lst)))
        for i, func_str in enumerate(lst, 1):
            print("\ni: {}, func_str: {}".format(i, func_str)) 


    def _construct_message(self):
        return "IT WORKS!"


if __name__ == "__main__":
    try:
        generate_complex_pictures = GenerateComplexPictures(
            n1=500,
            scale=16,
            scale_y=1.,
            x_offset=8,
            y_offset=8,
            delta=0.0001
        )

        generate_complex_pictures.do_calculations()

        generate_complex_pictures.save_new_z_function()
    except:
        generate_complex_pictures.delete_image_folder()
