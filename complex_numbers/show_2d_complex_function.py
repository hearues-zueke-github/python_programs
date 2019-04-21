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
import time
import traceback

import numpy as np

import matplotlib.pyplot as plt

from dotmap import DotMap

from indexed import IndexedOrderedDict
from collections import OrderedDict
from dotmap import DotMap
from PIL import Image, ImageDraw, ImageFont

import generate_generic_z_function

all_symbols_16 = np.array(list("0123456789ABCDEF"))
def get_random_string_base_16(n):
    l = np.random.randint(0, 16, (n, ))
    return "".join(all_symbols_16[l])

def get_date_time_str():
    dt = datetime.datetime.now()
    return "Y{:04}_m{:02}_d{:02}_H{:02}_M{:02}_S{:02}_f{:06}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
    # return "{:Y%Ym%md%dH%HM%MS%Sf%f}".format(datetime.datetime.now())

# TODO: create function with one of the founded function to get a higher resolution!
# TODO: generate gif animation of one variable changing e.g.
# TODO: generate better plots too!
class GenerateComplexPictures(Exception):
    def __init__(self,
                 modulo=10.,
                 
                 # n1_x=500,
                 # scale=16,
                 # scale_y=1.,
                 # x_offset=8,
                 # y_offset=8,

                 nx=None,
                 ny=None,
                 max_length=None,
                 x_center=None,
                 y_center=None,

                 delta=0.0001,
                 func_str=None,
                 main_folder="images/",
                 root_folder=None,
                 number=None):

        self.modulo = modulo
        self.delta = delta

        self.all_symbols_16 = np.array(list("0123456789ABCDEF"))
        self.all_symbols_64 = np.array(list(string.ascii_letters+string.digits+"-_"))

        self.file_extension_name = "_{}_{}".format(self.get_date_time_str(), self.get_random_string_base_16(16))

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
            raise Exception
            # self.n1_x = n1_x
            # self.scale = scale
            # self.scale_y = scale_y
            
            # self.x_offset = x_offset
            # self.y_offset = y_offset
            # self.n1_y = int(self.n1_x*self.scale_y)

            # self.xs1 = np.arange(0, self.n1_x)/self.n1_x*self.scale-self.x_offset+self.delta
            # self.ys1 = np.arange(0, self.n1_y)/self.n1_y*self.scale*self.scale_y-self.y_offset+self.delta

        self.ys1 = self.ys1[::-1]

        self.ys1_2d = np.zeros((self.ys1.shape[0], self.xs1.shape[0]))
        self.xs1_2d = self.ys1_2d.copy()

        self.xs1_2d[:] = self.xs1
        self.ys1_2d[:] = self.ys1.reshape((-1, 1))

        self.arr_xy_real_imag = np.vectorize(complex)(self.xs1_2d, self.ys1_2d)

        self.message = self._construct_message()
        super(GenerateComplexPictures, self).__init__(self.message)

        self.main_folder = main_folder
        if main_folder != "images/":
            main_folder += ("" if "/" == main_folder[-1] else "/")
            self.main_folder

        if root_folder == None:
            self.root_folder = ""
        else:
            self.root_folder = root_folder+("/" if root_folder[-1] != "/" else "")


        if number != None:
            self.number = number
            self.num_str = "_{:03}".format(number)

            self.z_func_file_name = "z_func{}.txt".format(self.num_str)
            self.path_folder_images = self.main_folder+self.root_folder+"z_funcs/"
            self.path_dir_arrs_data = self.main_folder+self.root_folder+"dm_objs/"
        else:
            self.number = None
            self.num_str = ""
            self.z_func_file_name = "z_func_{}.txt".format(self.file_extension_name)
            self.path_folder_images = self.main_folder+"{}{}/".format(
                self.root_folder,
                self.file_extension_name
            )

            self.path_dir_arrs_data = self.path_folder_images

        if not os.path.exists(self.path_folder_images):
            os.makedirs(self.path_folder_images)


        if func_str != None:
            self.func_str = func_str

            with open(self.path_folder_images+self.z_func_file_name, "w") as fout:
                fout.write(self.func_str)
        else:
            generate_generic_z_function.main(path_folder=self.path_folder_images, number=self.number, file_extension_name=self.file_extension_name)

            with open(self.path_folder_images+self.z_func_file_name, "r") as fin:
                self.func_str = fin.read()

        print("func_str: {}".format(self.func_str))
        self.func_str_complete = "lambda z: "+self.func_str

        self.f = self.get_f()

        print("self.path_folder_images: {}".format(self.path_folder_images))
        print("self.func_str_complete: {}".format(self.func_str_complete))

        # self.num_str = ""
        # if self.number != None:
        #     self.num_str = "_{:03}".format(self.number)


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
        dt = datetime.datetime.now()
        return "Y{:04}_m{:02}_d{:02}_H{:02}_M{:02}_S{:02}_f{:06}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
        # return "Y{}_m{}_d{}_H{}_M{}_S{}_f{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
        # return "{:%Y_%m_%d_%H_%M_%S_%f}".format(datetime.datetime.now())


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


    def calculate_arrs(self):
        print("Calculate self.arr_f and self.arr_scales")
        self.arr_f, self.arr_scales = np.vectorize(self.f)(self.arr_xy_real_imag)

        print("Calculate arr")
        arr_angle = np.angle(self.arr_f)
        idx = arr_angle < 0
        arr_angle[idx] = arr_angle[idx]+np.pi*2

        print("np.min(arr_angle): {}".format(np.min(arr_angle)))
        print("np.max(arr_angle): {}".format(np.max(arr_angle)))

        arr_angle_norm = arr_angle/(np.pi*2)
        print("np.min(arr_angle_norm): {}".format(np.min(arr_angle_norm)))
        print("np.max(arr_angle_norm): {}".format(np.max(arr_angle_norm)))

        print("Calculate abs")
        arr_abs = np.abs(self.arr_f)
        arr_abs_mod = arr_abs*self.arr_scales

        arr_abs_norm = arr_abs/np.max(arr_abs)
        arr_abs_mod_norm = arr_abs_mod/np.max(arr_abs_mod)


        print("Calculate x and y")
        f_norm_axis = lambda v: (lambda v2: v/np.max(v))(v-np.min(v))

        arr_x = self.arr_f.real
        arr_y = self.arr_f.imag
        arr_x_mod = arr_x*self.arr_scales
        arr_y_mod = arr_y*self.arr_scales

        arr_x_norm = f_norm_axis(arr_x)
        arr_y_norm = f_norm_axis(arr_y)
        arr_x_mod_norm = f_norm_axis(arr_x_mod)
        arr_y_mod_norm = f_norm_axis(arr_y_mod)


        self.arrs = IndexedOrderedDict([
            ("angle", arr_angle_norm),
            ("abs", arr_abs_norm),
            ("abs_mod", arr_abs_mod_norm),
            ("x", arr_x_norm),
            ("x_mod", arr_x_mod_norm),
            ("y", arr_y_norm),
            ("y_mod", arr_y_mod_norm),
        ])


    def calculate_rgb_pixs(self):
        print("Getting all data in pix_float array")
        shape = self.arr_xy_real_imag.shape+(3, )
        self.pixs1_dict = IndexedOrderedDict([
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

        self.pixs1_dict["f"][:, :, 0] = self.arrs["angle"]
        self.pixs1_dict["f"][:, :, 2] = self.arrs["abs"]*1/3+2/3

        self.pixs1_dict["f_mod"][:, :, 0] = self.arrs["angle"]
        self.pixs1_dict["f_mod"][:, :, 2] = self.arrs["abs_mod"]
        
        keys = ["angle", "abs", "abs_mod", "x", "x_mod", "y", "y_mod"]
        for key in keys:
            self.pixs1_dict[key][:, :, 0] = self.arrs[key]

        hsv_to_rgb_vectorized = np.vectorize(colorsys.hsv_to_rgb)

        print("Convert hsv to rgb and convert to uint8")

        self.pixs1_dict_rgb = IndexedOrderedDict()
        for key, value in self.pixs1_dict.items():
            pix_float_rgb = np.dstack(hsv_to_rgb_vectorized(value[..., 0], value[..., 1], value[..., 2]))
            self.pixs1_dict_rgb[key] = (pix_float_rgb*255.9).astype(np.uint8)

        self.pixs1 = [v for v in self.pixs1_dict_rgb.values()]

        self.imgs_orig = [Image.fromarray(pix.copy()) for pix in self.pixs1]


    def add_coordinate_lines(self):
        print("Adding coordinates x and y if possible")
        find_x = 0.
        find_y = 0.
        rows, cols = self.pixs1[0].shape[:2]
        line_col = np.argmin((self.xs1-find_x)**2)
        line_row = np.argmin((self.ys1-find_y)**2)

        if not(line_row < 0 or line_row >= rows):
            for pix in self.pixs1:
                pix[:, line_col] = pix[:, line_col]^(0xFF, )*3

        if not(line_col < 0 or line_col >= cols):
            for pix in self.pixs1:
                pix[line_row, :] = pix[line_row, :]^(0xFF, )*3


    def add_side_info(self):
        print("Creating the side infos")
        # this is needed for the info on the left side of the graph!
        pix2 = np.zeros((self.pixs1_dict_rgb["f"].shape[0], 300, 3), dtype=np.uint8)
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

        self.pixs2 = []
        for text_to_write in text_to_write_lst:
            img2_temp = img2.copy()
            d = ImageDraw.Draw(img2_temp)
            
            font_y_next_temp = font_y_next
            for text in text_to_write:
                d.text((8, font_y_next_temp), text, font=fnt, fill=(255, 255, 255))        
                font_y_next_temp += 24

            self.pixs2.append(np.array(img2_temp))

        self.imgs = []
        for pix2, pix1 in zip(self.pixs2, self.pixs1):
            pix3 = np.hstack((pix2, pix1))
            self.imgs.append(Image.fromarray(pix3))


    def save_all_images(self):
        self.suffixes = ["f", "f_mod", "angle", "abs", "abs_mod", "x", "x_mod", "y", "y_mod"]

        if self.root_folder != "":
            for suffix, img in zip(self.suffixes, self.imgs_orig):
                folder_path_f_orig = self.main_folder+""+self.root_folder+"orig_"+suffix+"/"
                if not os.path.exists(folder_path_f_orig):
                    os.makedirs(folder_path_f_orig)

                img.save(folder_path_f_orig+"{}_{}.png".format(suffix, self.num_str))
                # img.save(folder_path_f_orig+"{}{}_{}_{}.png".format(suffix, self.num_str,
                #             self.get_date_time_str(),
                #             self.get_random_string_base_16(16))
                # )

            for suffix, img in zip(self.suffixes, self.imgs):
                folder_path = self.main_folder+""+self.root_folder+"plot_"+suffix+"/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                img.save(folder_path+"{}_{}.png".format(suffix, self.num_str))
                # img.save(folder_path+suffix+"{}_{}_{}.png".format(
                #         self.num_str,
                #         self.get_date_time_str(),
                #         self.get_random_string_base_16(16)
                #     )
                # )
        else:
            # print("NO root_folder!!!!")
            for suffix, img in zip(self.suffixes, self.imgs_orig):
                folder_path_f_orig = self.main_folder+"plots_originals/orig_"+suffix+"/"
                if not os.path.exists(folder_path_f_orig):
                    os.makedirs(folder_path_f_orig)

                img.save(folder_path_f_orig+"{}{}{}.png".format(suffix, self.num_str,
                            self.file_extension_name)
                )

            for suffix, img in zip(self.suffixes, self.imgs):
                folder_path_f_orig = self.main_folder+"plots_with_side_info/orig_"+suffix+"/"
                if not os.path.exists(folder_path_f_orig):
                    os.makedirs(folder_path_f_orig)

                img.save(folder_path_f_orig+"{}{}{}.png".format(suffix, self.num_str,
                            self.file_extension_name)
                )

            for suffix, img in zip(self.suffixes, self.imgs):
                if self.number != None:
                    suffix += "_{}".format(self.number)
                img.save(self.path_folder_images+
                    "{}{}.png".format(suffix, self.file_extension_name)
                )


    def save_arrs_data(self):
        # if self.root_folder == "":
        if not os.path.exists(self.path_dir_arrs_data):
            os.makedirs(self.path_dir_arrs_data)
        print("self.path_dir_arrs_data: {}".format(self.path_dir_arrs_data))

        dm_obj = DotMap()

        dm_obj.func_str = self.func_str
        dm_obj.modulo = self.modulo
        dm_obj.delta = self.delta
        
        dm_obj.arr_xy_real_imag = self.arr_xy_real_imag
        dm_obj.arr_f = self.arr_f
        dm_obj.arr_scales = self.arr_scales

        dm_obj.nx = self.nx
        dm_obj.ny = self.ny
        dm_obj.max_length = self.max_length
        dm_obj.x_center = self.x_center
        dm_obj.y_center = self.y_center

        if self.number != None:
            self.path_file_arrs = self.path_dir_arrs_data+"dm_obj{}.pkl.gz".format(self.num_str)
        else:
            self.path_file_arrs = self.path_dir_arrs_data+"dm_obj{}.pkl.gz".format(self.file_extension_name)
        print("self.path_file_arrs: {}".format(self.path_file_arrs))
        with gzip.open(self.path_file_arrs, "wb") as fout:
            dill.dump(dm_obj, fout)


    # TODO: make it so that you can call this functions from outside too
    # with the self object! So that an extern z func array can be ploted too!
    def do_calculations(self):
        self.calculate_arrs()

        self.calculate_rgb_pixs()

        self.add_coordinate_lines()

        self.add_side_info()

        self.save_all_images()

        self.save_arrs_data()


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
            # n1_x=500,
            # scale=10,
            # scale_y=1,
            # x_offset=5,
            # y_offset=5,
            
            nx=480,
            ny=270,
            max_length=10,
            x_center=0.,
            y_center=0.,
            
            delta=0.0001,
            # func_str="z**2+z*4+complex(-10, 1)",
            modulo=1.,
        )
        try:
        # if True:
            generate_complex_pictures.do_calculations()
            # generate_complex_pictures.save_new_z_function()
            # break
        except KeyboardInterrupt:
            generate_complex_pictures.delete_image_folder()
            break
        except:
            # traceback.print_last()
            generate_complex_pictures.delete_image_folder()
            # sys.exit(-1234)


def do_only_gif_part(main_folder, root_folder):
    if "/" != main_folder[-1]:
        main_folder += "/"
    
    if "/" != root_folder[-1]:
        root_folder += "/"

    complete_path = main_folder+root_folder

    for i, (root_dir, dirs, files) in enumerate(os.walk(complete_path)):
        if len(dirs) == 0:
            continue

        print("\ni: {}".format(i))
        print("root_dir: {}".format(root_dir))
        # print("files: {}".format(files))
        print("dirs: {}".format(dirs))
        
        file_str_extension = "{}_{}".format(get_date_time_str(), get_random_string_base_16(16))
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

            dir_name_path = "images/gifs_{}/".format(dir_name)
            if not os.path.exists(dir_name_path):
                os.makedirs(dir_name_path)

            shutil.copy(gif_file_path, dir_name_path+"{}_{}.gif".format(dir_name, file_str_extension))



def create_gif_images():
    path_folder_data = "data/"
    if not os.path.exists(path_folder_data):
        os.makedirs(path_folder_data)

    z_func_template_raw = "np.sin(z.real*{})*z+complex(np.cos(z.real*{}+complex(z, complex(z.imag, z.real*3)).imag+{{}})+np.sin(z.imag+{{}}), np.sin(np.cos(z.real*{})+z.imag*{{}})+{{}}*z)*z*2"
    
    z_func_template = z_func_template_raw.format(
        1+np.random.random()*2,
        1+np.random.random()*3,
        1+np.random.random()*4
    )

    # z_func_template = "complex(z.real*np.sin({}+z.imag)+z.imag*{}+np.sin(z.imag), z.real*np.cos(z.imag*{}+1)+z.real**2*{}+1)"
    # z_func_template = "complex(z.real*{}+z.imag*{}+np.sin(z.imag), z.real*z.imag*{}+z.real**2*{}+1)"
    # z_func_template = "complex(np.sin(z.real+{})+z.imag*z**(1+{}), np.abs(np.sin(z.imag)+z.real))*complex(z*np.sin(z.imag+{}), z.imag*np.sin(np.cos(z.real)+{}))"
    # z_func_template = "z**(1+{}*0.1+0.1)*complex(z.real*({}+{}+1), z.imag*np.sin(z.real+{}))"
    # z_func_template = "z*complex(2+z.real*{}*0.25, 1+z.imag*{})*complex(2+z.imag*{}*0.25, 1+z.real*{})"

    z_funcs = []
    n1 = 5 # 9
    n2 = 5 # 9

    xs1 = []
    ys1 = []
    xs2 = []
    ys2 = []

    for i in range(0, n1):
        for j in range(0, n2):
            t1 = (i+(j%n2)/n2)/n1
            t2 = j/n2

            t1 *= np.pi*2
            t2 *= np.pi*2

            x1 = np.cos(t1)
            y1 = np.sin(t1)
            x2 = np.cos(t2)*1/n2
            y2 = np.sin(t2)*1/n2

            xs1.append(x1)
            ys1.append(y1)
            xs2.append(x2)
            ys2.append(y2)

            z_funcs.append(z_func_template.format(y2, x1, x2, y1))

    nx = 480
    ny = 270
    modulo = 6.
    max_len = 9.

    root_folder = "aa_nx_{}_ny_{}_mod_{}_max_len_{}_{}".format(
        nx,
        ny,
        "{:0.03f}".format(modulo).replace(".", "_"),
        "{:0.03f}".format(max_len).replace(".", "_"),
        get_random_string_base_16(4),
    )

    # save the z funcs into a file
    with open(path_folder_data+root_folder+".txt", "w") as fout:
        for z_func in z_funcs:
            fout.write(z_func+"\n")

    # for number, func_str in enumerate(z_funcs):
    for number, func_str in enumerate(z_funcs[:0]):
        print("\nDoing:")
        print("func_str: {}".format(func_str))
        generate_complex_pictures = GenerateComplexPictures(
            # n1_x=640,
            # scale=10,
            # scale_y=0.75,
            # x_offset=5,
            # y_offset=3.75,
            
            nx=nx,
            ny=ny,
            max_length=max_len,
            x_center=0.,
            y_center=0.,

            modulo=modulo,

            delta=0.0001,
            func_str=func_str,
            root_folder=root_folder,
            number=number,
            # func_str="z*complex(z.imag, z.real*1.2)"
        )
        try:
            generate_complex_pictures.do_calculations()
            # break
            # generate_complex_pictures.save_new_z_function()
        except:
            generate_complex_pictures.delete_image_folder()
            traceback.print_last()
            sys.exit(-1234)

    root_folder = "aa_nx_480_ny_270_mod_6_000_max_len_9_000_ABFC"
    do_only_gif_part("images/", root_folder)


def interpolate_images():
    main_folder = "interpolations/"

    file_paths = []
    for root_dir, dirs, files in os.walk(main_folder+"dm_objs/"):
        for file_name in files:
            if not ".pkl.gz" in file_name:
                continue
            file_paths.append(root_dir+file_name)
        break

    # files = sorted(files)
    file_paths = sorted(file_paths)

    dm_objs = []
    for file_path in file_paths:
        print("file_path: {}".format(file_path))
        with gzip.open(file_path, "rb") as fin:
            dm_obj = dill.load(fin)
        dm_objs.append(dm_obj)

    for i, dm_obj in enumerate(dm_objs):
        print("\ni: {}".format(i))
        print("  dm_obj.keys(): {}".format(dm_obj.keys()))
        # print("  dm_obj.arr_xy_real_imag.shape: {}".format(dm_obj.arr_xy_real_imag.shape))
        # print("  dm_obj.arr_f.shape: {}".format(dm_obj.arr_f.shape))
        # print("  dm_obj.arr_scales.shape: {}".format(dm_obj.arr_scales.shape))
        # print("  dm_obj.delta: {}".format(dm_obj.delta))
        # print("  dm_obj.modulo: {}".format(dm_obj.modulo))

    root_folder = "combined_images_{}_{}/".format(get_date_time_str(), get_random_string_base_16(4))
    if not os.path.exists(main_folder+root_folder):
        os.makedirs(main_folder+root_folder)

    dm_obj_new = DotMap()
    dm_obj_new.arr_f
    # TODO: can be finished easily!

    modulo = 4.

    random_idxs = np.random.permutation(np.arange(0, len(dm_objs)))[:np.random.randint(3, len(dm_objs))]
    random_idxs = np.hstack((random_idxs, [random_idxs[0]]))
    print("random_idxs[:2]: {}".format(random_idxs[:2]))

    with gzip.open(main_folder+root_folder+"random_idxs.pkl.gz", "wb") as fout:
        dill.dump(random_idxs, fout)
    with open(main_folder+root_folder+"random_idxs.txt", "w") as fout:
        fout.write(", ".join(list(map(str, random_idxs)))+"\n")

    print("random_idxs: {}".format(random_idxs))
    time.sleep(2)
    # return

    # func_str_1 = dm_objs[random_idxs[0]].func_str
    # func_str_2 = dm_objs[random_idxs[1]].func_str

    # print("func_str_1: {}".format(func_str_1))
    # print("func_str_2: {}".format(func_str_2))

    # TODO: make alphas mor smooth and acceleration!
    amount_interpolations = 20
    alphas = (lambda x: x/x[-1])(np.cumsum(np.sin(np.arange(0, (amount_interpolations+1))/(amount_interpolations+1)*np.pi)+0.1))
    # alphas = np.arange(0, amount_interpolations)/amount_interpolations
    with open(main_folder+root_folder+"alphas.txt", "w") as fout:
        fout.write(", ".join(list(map(str, alphas)))+"\n")

    kwargs = {
        'modulo': modulo,

        'nx': dm_objs[0].nx,
        'ny': dm_objs[0].ny,
        'max_length': dm_objs[0].max_length,
        'x_center': dm_objs[0].x_center,
        'y_center': dm_objs[0].y_center,
 
        'delta': 0.0001,
        'func_str': '',
        'main_folder': main_folder,
        'root_folder': root_folder,
        'number': i,
    }

    number = 0
    for j in range(0, random_idxs.shape[0]-1):
        func_str_1 = dm_objs[random_idxs[j]].func_str
        func_str_2 = dm_objs[random_idxs[j+1]].func_str

        for i, alpha in enumerate(alphas, 0):
            func_str = "(1-{})*({})+{}*({})".format(
                alpha,
                func_str_1,
                alpha,
                func_str_2,
            )

            # generate_complex_pictures = GenerateComplexPictures(
            #              modulo=modulo,

            #              nx=dm_objs[0].nx,
            #              ny=dm_objs[0].ny,
            #              max_length=dm_objs[0].max_length,
            #              x_center=dm_objs[0].x_center,
            #              y_center=dm_objs[0].y_center,

            #              delta=0.0001,
            #              func_str=func_str,
            #              main_folder=main_folder,
            #              root_folder=root_folder,
            #              number=i)

            kwargs['func_str'] = func_str
            kwargs['number'] = i+number
            generate_complex_pictures = GenerateComplexPictures(**kwargs)


            try:
                generate_complex_pictures.do_calculations()
                # break
                # generate_complex_pictures.save_new_z_function()
            except:
                generate_complex_pictures.delete_image_folder()
                traceback.print_last()
                sys.exit(-1234)

        number += i+1

    # GenerateComplexPictures()
    print("No gifs created!! So far!")
    # do_only_gif_part(main_folder, root_folder)


if __name__ == "__main__":
    create_random_z_funcs()
    # create_gif_images()
    # interpolate_images()
