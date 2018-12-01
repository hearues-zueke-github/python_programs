#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import colorsys
import datetime
import dill
import gzip
import os
import shutil
import string
import subprocess
import sys
import traceback

import numpy as np

import matplotlib.pyplot as plt

from indexed import IndexedOrderedDict
from collections import OrderedDict
from dotmap import DotMap
from PIL import Image, ImageDraw, ImageFont

import generate_generic_z_function

# TODO: create function with one of the founded function to get a higher resolution!
# TODO: generate gif animation of one variable changing e.g.
# TODO: generate better plots too!
class GenerateComplexPictures(Exception):
    def __init__(self,
                 modulo=10.,
                 n1_x=500,
                 scale=16,
                 scale_y=1.,
                 x_offset=8,
                 y_offset=8,

                 nx=None,
                 ny=None,
                 max_length=None,
                 x_center=None,
                 y_center=None,

                 delta=0.0001,
                 func_str=None,
                 root_folder=None,
                 number=None):

        self.modulo = modulo
        self.delta = delta

        # TODO: need a big fix!
        if nx!=None and ny!=None and max_length!=None and x_center!=None and y_center!=None:
            self.nx = nx
            self.ny = ny
            self.max_length = max_length

            scale_x = max_length
            scale_y = max_length
            if nx > ny:
                scale_y = max_length*ny/nx
            else:
                scale_x = max_length*nx/ny

            self.scale_x = scale_x
            self.scale_y = scale_y
            
            self.x_center = x_center
            self.y_center = y_center

            self.xs1 = np.arange(0, self.nx)/self.nx*self.scale_x-self.scale_x/2+self.x_center+self.delta
            self.ys1 = np.arange(0, self.ny)/self.ny*self.scale_y-self.scale_y/2+self.y_center+self.delta

            self.x_min = self.x_center-self.scale_x/2
            self.x_max = self.x_center+self.scale_x/2
            self.y_min = self.y_center-self.scale_y/2
            self.y_max = self.y_center+self.scale_y/2

            # globals()["xs1"] = self.xs1
            # globals()["ys1"] = self.ys1

            # print("\nself.nx: {}".format(self.nx))
            # print("self.ny: {}".format(self.ny))

            # print("\nself.max_length: {}".format(self.max_length))
            # print("self.scale_x: {}".format(self.scale_x))
            # print("self.scale_y: {}".format(self.scale_y))
            # print("self.x_center: {}".format(self.x_center))
            # print("self.y_center: {}".format(self.y_center))

            # print("TEST!")
            # sys.exit(-10)
        else:
            self.n1_x = n1_x
            self.scale = scale
            self.scale_y = scale_y
            
            self.x_offset = x_offset
            self.y_offset = y_offset
            self.n1_y = int(self.n1_x*self.scale_y)

            self.xs1 = np.arange(0, self.n1_x)/self.n1_x*self.scale-self.x_offset+self.delta
            self.ys1 = np.arange(0, self.n1_y)/self.n1_y*self.scale*self.scale_y-self.y_offset+self.delta

        self.ys1 = self.ys1[::-1]

        self.ys1_2d = np.zeros((self.ys1.shape[0], self.xs1.shape[0]))
        self.xs1_2d = self.ys1_2d.copy()

        self.xs1_2d[:] = self.xs1 
        self.ys1_2d[:] = self.ys1.reshape((-1, 1))

        self.arr_complex = np.vectorize(complex)(self.xs1_2d, self.ys1_2d)

        self.all_symbols_16 = np.array(list("0123456789ABCDEF"))
        self.all_symbols_64 = np.array(list(string.ascii_letters+string.digits+"-_"))

        self.message = self._construct_message()
        super(GenerateComplexPictures, self).__init__(self.message)

        if root_folder == None:
            self.root_folder = ""
        else:
            self.root_folder = root_folder+("/" if root_folder[-1] != "/" else "")


        if number != None:
            self.number = number

            self.z_func_file_name = "z_func_{}.txt".format(self.number)
            self.path_folder_images = "images/"+self.root_folder+"z_funcs/"
        else:
            self.number = None
            self.z_func_file_name = "z_func.txt"
            self.path_folder_images = "images/{}{}_{}/".format(
                self.root_folder,
                self.get_date_time_str(),
                self.get_random_string_base_16(16)
            )

        if not os.path.exists(self.path_folder_images):
            os.makedirs(self.path_folder_images)


        if func_str != None:
            self.func_str = func_str

            with open(self.path_folder_images+self.z_func_file_name, "w") as fout:
                fout.write(self.func_str)
        else:
            generate_generic_z_function.main(path_folder=self.path_folder_images, number=self.number)

            with open(self.path_folder_images+self.z_func_file_name, "r") as fin:
                self.func_str = fin.read()

        print("func_str: {}".format(self.func_str))
        self.func_str_complete = "lambda z: "+self.func_str

        self.f = self.get_f()

        print("self.path_folder_images: {}".format(self.path_folder_images))
        print("self.func_str_complete: {}".format(self.func_str_complete))
        # sys.exit(-2)



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
        arr_angle[idx] = arr_angle[idx]+np.pi*2

        print("np.min(arr_angle): {}".format(np.min(arr_angle)))
        print("np.max(arr_angle): {}".format(np.max(arr_angle)))

        arr_angle_norm = arr_angle/(np.pi*2)
        print("np.min(arr_angle_norm): {}".format(np.min(arr_angle_norm)))
        print("np.max(arr_angle_norm): {}".format(np.max(arr_angle_norm)))

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


        arrs = {
            "angle": arr_angle_norm,
            "abs": arr_abs_norm,
            "abs_mod": arr_abs_mod_norm,
            "x": arr_x_norm,
            "x_mod": arr_x_mod_norm,
            "y": arr_y_norm,
            "y_mod": arr_y_mod_norm,
        }


        print("Getting all data in pix_float array")
        shape = arr_complex.shape+(3, )
        pixs1_dict = IndexedOrderedDict([
            ("f", np.ones(shape)),
            ("f_mod", np.ones(shape)),
            ("angle", np.ones(shape)),
            ("abs", np.ones(shape)),
            ("abs_mod", np.ones(shape)),
            ("x", np.ones(shape)),
            ("x_mod", np.ones(shape)),
            ("y", np.ones(shape)),
            ("y_mod", np.ones(shape)),
        ])

        pixs1_dict["f"][:, :, 0] = arrs["angle"]
        pixs1_dict["f"][:, :, 2] = arrs["abs"] # *1/3+2/3

        pixs1_dict["f_mod"][:, :, 0] = arrs["angle"]
        pixs1_dict["f_mod"][:, :, 2] = arrs["abs_mod"] # *1/3+2/3
        
        keys = ["angle", "abs", "abs_mod", "x", "x_mod", "y", "y_mod"]
        for key in keys:
            pixs1_dict[key][:, :, 0] = arrs[key]

        hsv_to_rgb_vectorized = np.vectorize(colorsys.hsv_to_rgb)

        print("Convert hsv to rgb and convert to uint8")

        pixs1_dict_rgb = IndexedOrderedDict()
        for key, value in pixs1_dict.items():
            pix_float_rgb = np.dstack(hsv_to_rgb_vectorized(value[..., 0], value[..., 1], value[..., 2]))
            pixs1_dict_rgb[key] = (pix_float_rgb*255.9).astype(np.uint8)

        pixs1 = [v for v in pixs1_dict_rgb.values()]

        imgs_orig = [Image.fromarray(pix.copy()) for pix in pixs1]

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
        pix2 = np.zeros((pixs1_dict_rgb["f"].shape[0], 300, 3), dtype=np.uint8)
        img2 = Image.fromarray(pix2)
        
        # get a font
        fnt = ImageFont.truetype('monofonto.ttf', 16)
        d = ImageDraw.Draw(img2)
        d.text((8, 8), "Used function:", font=fnt, fill=(255, 255, 255))

        func_str_split = [self.func_str_complete[30*i:30*(i+1)] for i in range(0, len(self.func_str_complete)//30+1)]
        for i, func_str_part in enumerate(func_str_split, 1):
            d.text((8, 8+24*i), func_str_part, font=fnt, fill=(255, 255, 255))
        
        font_y_next = 8+24*(i+2)
        d.text((8, font_y_next), "x_min: {:3.02f}, x_max: {:3.02f}".format(self.x_min, self.x_max), font=fnt, fill=(255, 255, 255))        
        # d.text((8, font_y_next), "x_min: {:3.02f}, x_max: {:3.02f}".format(-self.x_offset, self.scale-self.x_offset), font=fnt, fill=(255, 255, 255))        
        
        font_y_next += 24
        d.text((8, font_y_next), "y_min: {:3.02f}, y_max: {:3.02f}".format(self.y_min, self.y_max), font=fnt, fill=(255, 255, 255))        
        # d.text((8, font_y_next), "y_min: {:3.02f}, y_max: {:3.02f}".format(-self.y_offset, self.scale*self.scale_y-self.y_offset), font=fnt, fill=(255, 255, 255))        
        
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
        
        num_str = ""
        if self.number != None:
            num_str = "_{:03}".format(self.number)

        if self.root_folder != "":
            for suffix, img in zip(suffixes, imgs_orig):
                folder_path_f_orig = "images/"+self.root_folder+suffix+"_orig/"
                if not os.path.exists(folder_path_f_orig):
                    os.makedirs(folder_path_f_orig)

                img.save(folder_path_f_orig+"{}{}_{}_{}.png".format(suffix, num_str,
                            self.get_date_time_str(),
                            self.get_random_string_base_16(16))
                )

            for suffix, img in zip(suffixes, imgs):
                folder_path = "images/"+self.root_folder+suffix+"/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                img.save(folder_path+suffix+"{}_{}_{}.png".format(
                        num_str,
                        self.get_date_time_str(),
                        self.get_random_string_base_16(16)
                    )
                )
        else:
            for suffix, img in zip(suffixes, imgs_orig):
                folder_path_f_orig = "images/originals/"+suffix+"_orig/"
                if not os.path.exists(folder_path_f_orig):
                    os.makedirs(folder_path_f_orig)

                img.save(folder_path_f_orig+"{}{}_{}_{}.png".format(suffix, num_str,
                            self.get_date_time_str(),
                            self.get_random_string_base_16(16))
                )

            for suffix, img in zip(suffixes, imgs):
                folder_path_f_orig = "images/plots/"+suffix+"_orig/"
                if not os.path.exists(folder_path_f_orig):
                    os.makedirs(folder_path_f_orig)

                img.save(folder_path_f_orig+"{}{}_{}_{}.png".format(suffix, num_str,
                            self.get_date_time_str(),
                            self.get_random_string_base_16(16))
                )

            for suffix, img in zip(suffixes, imgs):
                if self.number != None:
                    suffix += "_{}".format(self.number)
                file_name = "_{}.png".format(suffix)
                img.save(self.path_folder_images+
                    "{}_{}".format(
                        self.get_date_time_str(),
                        self.get_random_string_base_16(16)
                    )+
                    file_name
                )


    def save_new_z_function(self):
        func_str_complete = self.func_str_complete
        
        path_folder_data = "data/"
        if not os.path.exists(path_folder_data):
            os.makedirs(path_folder_data)

        path_file_data = path_folder_data+"working_z_functions.pkl.gz"
        path_file_data_txt = path_folder_data+"working_z_functions.txt"

        if not os.path.exists(path_file_data):
            data = DotMap()
            data.func_str_lst = [func_str_complete]

            with gzip.open(path_file_data, "wb") as fout:
                dill.dump(data, fout)
        else:
            with gzip.open(path_file_data, "rb") as fin:
                data = dill.load(fin)

            data.func_str_lst.append(func_str_complete)
            
            with open(path_file_data_txt, "w") as fout:
                for line in data.func_str_lst:
                    fout.write(line+"\n")

            with gzip.open(path_file_data, "wb") as fout:
                dill.dump(data, fout)

        print("newest founded function:\n{}".format(func_str_complete))

        # print("data.func_str_lst:\n{}".format(data.func_str_lst))
        # lst = data.func_str_lst

        # print("Amount of found functions: {}".format(len(lst)))
        # for i, func_str in enumerate(lst, 1):
        #     print("\ni: {}, func_str: {}".format(i, func_str)) 


    def _construct_message(self):
        return "IT WORKS!"


def create_random_z_funcs():
    for i in range(0, 100):
        print("\ni: {}".format(i))
        generate_complex_pictures = GenerateComplexPictures(
            n1_x=500,
            scale=10,
            scale_y=1,
            x_offset=5,
            y_offset=5,
            delta=0.0001,
            # func_str="z**2+z*4+complex(-10, 1)",
            modulo=1.,
        )
        try:
        # if True:
            generate_complex_pictures.do_calculations()
            # generate_complex_pictures.save_new_z_function()
            # break
        except:
            generate_complex_pictures.delete_image_folder()


def create_gif_images():
    path_folder_data = "data/"
    if not os.path.exists(path_folder_data):
        os.makedirs(path_folder_data)

    z_func_template = "complex(z.real*np.sin({}+z.imag)+z.imag*{}+np.sin(z.imag), z.real*np.cos(z.imag*{}+1)+z.real**2*{}+1)"
    # z_func_template = "complex(z.real*{}+z.imag*{}+np.sin(z.imag), z.real*z.imag*{}+z.real**2*{}+1)"
    # z_func_template = "complex(np.sin(z.real+{})+z.imag*z**(1+{}), np.abs(np.sin(z.imag)+z.real))*complex(z*np.sin(z.imag+{}), z.imag*np.sin(np.cos(z.real)+{}))"
    # z_func_template = "z**(1+{}*0.1+0.1)*complex(z.real*({}+{}+1), z.imag*np.sin(z.real+{}))"
    # z_func_template = "z*complex(2+z.real*{}*0.25, 1+z.imag*{})*complex(2+z.imag*{}*0.25, 1+z.real*{})"

    z_funcs = []
    # n = 17
    n = 9
    n1_x = 9

    xs1 = []
    ys1 = []
    xs2 = []
    ys2 = []

    for i in range(0, n): # TODO: can be made out of two variables too!
        for j in range(0, n1_x):
            t1 = (i+(j%n1_x)/n1_x)/n
            t2 = j/n1_x

            # print("i: {}, t1: {:0.04f}, t2: {:0.04f}".format(i, t1, t2))

            t1 *= np.pi*2
            t2 *= np.pi*2

            x1 = np.cos(t1)
            y1 = np.sin(t1)
            x2 = np.cos(t2)*1/n1_x
            y2 = np.sin(t2)*1/n1_x

            xs1.append(x1)
            ys1.append(y1)
            xs2.append(x2)
            ys2.append(y2)

            # print("x1: {:0.03f}, y1: {:0.03f}, x2: {:0.03f}, y2: {:0.03f}".format(x1, y1, x2, y2))

            # z_funcs.append(z_func_template.format(y2, x1*0.5+0.5))#, x2, y1))
            z_funcs.append(z_func_template.format(y2, x1, x2, y1))
            # z_funcs.append(z_func_template.format(y2, x1*0.5+0.5, x2, y1))

    # x1 = 1
    # y1 = 0
    # x2 = 1/n1_x
    # y2 = 0

    # xs1.append(x1)
    # ys1.append(y1)
    # xs2.append(x2)
    # ys2.append(y2)

    # print("x1: {:0.03f}, y1: {:0.03f}, x2: {:0.03f}, y2: {:0.03f}".format(x1, y1, x2, y2))

    # plt.figure()

    # ts = np.arange(0, len(xs1))
    # p1 = plt.plot(ts, xs1, "b.-")[0]
    # p2 = plt.plot(ts, ys1, "g.-")[0]
    # p3 = plt.plot(ts, xs2, "r.-")[0]
    # p4 = plt.plot(ts, ys2, "k.-")[0]

    # plt.legend((p1, p2, p3, p4), ("xs1", "ys1", "xs2", "ys2"))

    # plt.show()

    # sys.exit(-123)

    # z_funcs.append(z_func_template.format(y2, x1*0.5+0.5))#, x2, y1))
    z_funcs.append(z_func_template.format(y2, x1, x2, y1))

    # sys.exit(-1)
    root_folder = "aa_test12_final"

    # save the z funcs into a file
    with open(path_folder_data+root_folder+".txt", "w") as fout:
        for z_func in z_funcs:
            fout.write(z_func+"\n")

    for number, func_str in enumerate(z_funcs):
    # for number, func_str in enumerate(z_funcs[:0]):
        print("\nDoing:")
        print("func_str: {}".format(func_str))
        try:
            generate_complex_pictures = GenerateComplexPictures(
                # n1_x=640,
                # scale=10,
                # scale_y=0.75,
                # x_offset=5,
                # y_offset=3.75,
                
                nx=480,
                ny=270,
                max_length=5.,
                x_center=0.,
                y_center=0.,

                modulo=1,

                delta=0.0001,
                func_str=func_str,
                root_folder=root_folder,
                number=number,
                # func_str="z*complex(z.imag, z.real*1.2)"
            )
            generate_complex_pictures.do_calculations()
            # generate_complex_pictures.save_new_z_function()
        except:
            generate_complex_pictures.delete_image_folder()
            traceback.print_last()
            sys.exit(-1234)


    complete_path = "images/"+root_folder+"/"

    for i, (root_dir, dirs, files) in enumerate(os.walk(complete_path)):
        if len(dirs) == 0:
            continue

        print("\ni: {}".format(i))
        print("root_dir: {}".format(root_dir))
        # print("files: {}".format(files))
        print("dirs: {}".format(dirs))
        
        root_dir = root_dir
        for dir_name in dirs:
            # if not "orig" in root_dir:
            if not "orig" in dir_name:
                continue

            dir_path = root_dir+dir_name+"/"

            print("dir_path: {}".format(dir_path))

            process = subprocess.Popen(['../../../create_gif.sh'], cwd=dir_path)
            process.wait()

            print("finished with '{}'".format(dir_path))

            gif_file_path = root_dir+dir_name+".gif"
            shutil.move(dir_path+"myimage.gif", gif_file_path)

            dir_name_path = "images/{}_gifs/".format(dir_name)
            if not os.path.exists(dir_name_path):
                os.makedirs(dir_name_path)

            shutil.copy(gif_file_path, dir_name_path+"{}_{}.gif".format(dir_name, generate_complex_pictures.get_random_string_base_16(16)))


if __name__ == "__main__":
    # create_random_z_funcs()
    create_gif_images()
