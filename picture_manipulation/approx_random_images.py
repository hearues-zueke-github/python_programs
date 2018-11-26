#! /usr/bin/python3.6

import dill
import os
import pdb
import shutil
import string
import sys

import numpy as np

from dotmap import DotMap
from PIL import Image

def get_random_64_bit_number(n):
    s = string.ascii_lowercase+string.ascii_uppercase+string.digits+"-_"
    arr = np.array(list(s))
    return "".join(np.random.choice(arr, (n, )).tolist())

class BitNeighborManipulation(Exception):
    def __init__(self, ft=2, with_frame=True, path_lambda_functions=None):
        self.ft = ft
        if with_frame:
            self.add_frame = self._get_add_frame_function() # function
            self.remove_frame = self._get_remove_frame_function() # function
        else:
            self.add_frame = None
            self.remove_frame = None

        self.max_bit_operators = 4
        self.bit_operators_idx = [0, 1, 2, 3]
        
        self.get_pixs = self._generate_pixs_function() # function
        self.bit_operations = self._generate_lambda_functions(path_lambda_functions) # list of lambdas
        self.it1 = 0 # for the iterator variable (1st)
        self.it2 = 0 # for the iterator variable (2nd)

    def _get_add_frame_function(self):
        ft = self.ft
        def add_frame(pix_bw):
            t = np.vstack((pix_bw[-ft:], pix_bw, pix_bw[:ft]))
            return np.hstack((t[:, -ft:], t, t[:, :ft]))
        return add_frame

    def _generate_pixs_function(self):
        ft = self.ft

        def generate_pixs(pix_bw):
            height, width = pix_bw.shape[:2]

            zero_row = np.zeros((width, ), dtype=np.uint8)
            zero_col = np.zeros((height, 1), dtype=np.uint8)

            move_arr_u = lambda pix_bw: np.vstack((pix_bw[1:], zero_row))
            move_arr_d = lambda pix_bw: np.vstack((zero_row, pix_bw[:-1]))
            move_arr_l = lambda pix_bw: np.hstack((pix_bw[:, 1:], zero_col))
            move_arr_r = lambda pix_bw: np.hstack((zero_col, pix_bw[:, :-1]))

            pixs = np.zeros((ft*2+1, ft*2+1, height, width), dtype=np.uint8)
            pixs[ft, ft] = pix_bw

            # first set all y pixs (center ones)
            for i in range(ft, 0, -1):
                pixs[i-1, ft] = move_arr_u(pixs[i, ft])
            for i in range(ft, ft*2):
                pixs[i+1, ft] = move_arr_d(pixs[i, ft])

            # then set all x pixs (except the center ones, they are already set)
            for j in range(0, ft*2+1):
                for i in range(ft, 0, -1):
                    pixs[j, i-1] = move_arr_l(pixs[j, i])
                for i in range(ft, ft*2):
                    pixs[j, i+1] = move_arr_r(pixs[j, i])

            return pixs

        return generate_pixs

    def _generate_lambda_functions(self, path_lambda_functions):

        if not os.path.exists(path_lambda_functions):
            print("File path '{}' does not exists!".fornat(path_lambda_functions))
            print("Will use default lambda functions then instead!")
            sys.exit(-1)

        with open(path_lambda_functions, "r") as fin:
            # lines = fin.readlines()
            lines = list(filter(lambda x: len(x) > 0, fin.read().splitlines()))

        # TODO: check every single line, if it is matching with the variable convention!
        # TODO: add a security function, where each line will be checked up
        lambdas = [eval(line) for line in lines]
        # self.max_bit_operators = 5
        self.max_bit_operators = len(lambdas)
        # pdb.set_trace()

        return lambdas

        # TODO: make this as the default lambdas list!
        return [
        lambda:
        p_ulll,
        lambda:
        p_ul,
        lambda:
        ((p_ull&p_ddr)&(p==0))|
        ((p_ul&p_ddrr&p_dl)&(p==1)),
        lambda: \
        ((p_uu&p_dd&p_rr&p_ll|p_u&p_d&p_l&p_r)&(p==0))|
        ((p_uull&p_uurr&p_ddll&p_ddrr|p_ul&p_ur&p_dl&p_dr)&(p==1)),
        lambda: \
        ((p_u&p_r|p_ur&p_dr)&(p==0))|
        ((p_d&p_l|p_ul&p_dl)&(p==1)),
        lambda: \
        ((p_u&p_l)&(p==0))^
        ((p_d&p_r)&(p==1)),
        lambda: \
        ((p_u&p_l|p_d&p_r)&(p==1))^
        ((p_u&p_d|p_l&p_r)&(p==0))^
        ((p_u&p_d&p_l&p_r)),
        lambda: \
        ((p_u&p_l|p_d&p_r)&(p==0))^
        ((p_u&p_d|p_l&p_r)&(p==1))^
        ((p_u&p_d&p_l&p_r)),
        lambda: \
        ((p_u&p_r|p_d&p_l)&(p==1))^
        ((p_u&p_l|p_l&p_d)&(p==0))^
        ((p_u&p_d&p_l&p_r)),
        lambda: \
        ((p_u&p_r|p_d&p_l)&(p==0))^
        ((p_u&p_l|p_r&p_d)&(p==1))^
        ((p_u&p_d&p_l&p_r)),
        lambda: \
        ((p_u&p_r&p_d|p_l)&(p==1))^
        ((p_u&p_r|p_l&p_d)&(p==0))^
        ((p_u&p_d&p_l&p_r)),
        lambda: \
        ((p_u&p_r&p_d|p_l)&(p==1))^
        ((p_u&p_r|p_l&p_d)&(p==0))^
        ((p_u&p_d&p_l&p_r))
        ]

    def _get_remove_frame_function(self):
        ft = self.ft
        def remove_frame(pix_bw):
            return pix_bw[ft:-ft, ft:-ft]
        return remove_frame

    def apply_neighbor_logic_1_bit(self, pix_bw):
        if self.add_frame != None:
            pix_bw = self.add_frame(pix_bw)
        pixs = self.get_pixs(pix_bw)

        ft = self.ft
        exec("globals()['p'] = pixs[ft, ft]")
        for y in range(0, ft*2+1):
            for x in range(0, ft*2+1):
                if y == ft and x == ft:
                    continue
                # Removed "p_" at the beginning of each variable name!
                var_name = ("u"*(ft-y) if y < ft else "d"*(y-ft))+("l"*(ft-x) if x < ft else "r"*(x-ft))
                # variables.append(var_name)
                exec("globals()['{}'] = pixs[{}, {}]".format(var_name, y, x))

        # idxs_lambda = (self.it1+self.it2)%self.max_bit_operators
        # idxs_lambda = (self.it1)%self.max_bit_operators
        # pix_bw1 = self.bit_operations[idxs_lambda]()
        # pix_bw1 = self.bit_operations[self.bit_operators_idx[(self.it1+self.it2)%self.max_bit_operators]]()
        pix_bw1 = self.bit_operations[(self.it1)%self.max_bit_operators]()
        # pix_bw2 = self.bit_operations[(self.it1+1)%self.max_bit_operators]()
        # pix_bw3 = self.bit_operations[self.bit_operators_idx[(self.it1+2)%self.max_bit_operators]]()

        # assert np.sum(pix_prev_1 != pix_prev_2) == 0
        pix_bw = pix_bw1
        # pix_bw = pix_bw1^pix_bw2
        # pix_bw = pix_bw1^pix_bw2^pix_bw3
        # pix_bw = self.bit_operations[self.bit_operators_idx[(self.it1+self.it2)%self.max_bit_operators]]()
        self.it2 += 1
        
        if self.remove_frame != None:
            return self.remove_frame(pix_bw)
        return pix_bw

    def apply_neighbor_logic(self, pix_bws):
        pix_bws_new = []

        self.it2 = 0
        for i, pix_bw in enumerate(pix_bws):
           pix_bws_new.append(self.apply_neighbor_logic_1_bit(pix_bw))
        self.it1 += 1

        return pix_bws_new


def create_1_bit_neighbour_pictures(height, width):
    path_pictures = "images/changing_bw_1_bit_{}_{}/".format(height, width)
    if not os.path.exists(path_pictures):
        os.system("rm -rf {}".format(path_pictures))
    if not os.path.exists(path_pictures):
        os.makedirs(path_pictures)

    pix_bw = np.random.randint(0, 2, (height, width), dtype=np.uint8)
    Image.fromarray(pix_bw*255).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, 0))

    # so long there are white pixels, repeat the elimination_process!
    it = 1
    pix_bw_prev = pix_bw.copy()
    pixs = [pix_bw.copy()]
    # repeat anything until it is complete blank / black / 0
    while np.sum(pix_bw == 1) > 0:
        print("it: {}".format(it))
        
        # TODO: need to be fixed!
        pix_bw = apply_neighbour_logic(pix_bw)

        Image.fromarray(pix_bw*255).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))
        it += 1

def create_1_byte_neighbour_pictures(height, width):
    path_pictures = "images/changing_bw_1_byte_{}_{}/".format(height, width)
    if not os.path.exists(path_pictures):
        os.system("rm -rf {}".format(path_pictures))
    if not os.path.exists(path_pictures):
        os.makedirs(path_pictures)

    get_pix_bw = lambda: np.random.randint(0, 2, (height, width), dtype=np.uint8)
    pix_bws = [get_pix_bw() for _ in range(0, 8)]

    def combine_1_bit_neighbours(pix_bws):
        pix = np.zeros(pix_bws[0].shape, dtype=np.uint8)
        for i, pix_bw in enumerate(pix_bws):
            pix += pix_bw<<i
        return pix

    pix_combine = combine_1_bit_neighbours(pix_bws)
    Image.fromarray(pix_combine).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, 0))

    # so long there are white pixels, repeat the elimination_process!
    it = 1
    # repeat anything until it is complete blank / black / 0
    while np.sum([np.sum(pix_bw == 1) for pix_bw in pix_bws]) > 0:
        print("it: {}".format(it))
        
        for i in range(0, 8):
            # TODO: need to be fixed!
            pix_bws[i] = apply_neighbour_logic(pix_bws[i])

        pix_combine = combine_1_bit_neighbours(pix_bws)
        Image.fromarray(pix_combine).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))
        it += 1

def create_3_byte_neighbour_pictures(img_type,
    height=None, width=None, same_image=None, with_frame=None,
    image_path=None, max_iterations=-1, path_lambda_functions=None,
    resize_params=None, ft=2, num_copies_first_image=3,
    amount_combines=1, gif_delay=5, fps_movie=20, folder_suffix="",
    height_resize=None, width_resize=None):
    prev_folder = os.getcwd()

    get_pix_bws_from_pix_img = lambda pix_img: [(pix_c>>j)&0x1 for pix_c in [pix_img[:, :, i] for i in range(0, 3)] for j in range(0, 8)]
    
    path_suffix = ("" if folder_suffix == "" else "_"+folder_suffix)

    if img_type == "picture":
        if image_path == None:
            sys.exit(-1)

        if not os.path.exists(image_path):
            print("Path to image '{}' does not exists!".format(image_path))
            return -1

        img = Image.open(image_path)
        pix_img = np.array(img)
        height, width = pix_img.shape[:2]

        path_pictures = "images/changing_image_{}_{}{}/".format(height, width, path_suffix)
        
    elif img_type == "random":
        if height == None or \
           width == None or \
           same_image == None or \
           with_frame == None:
            system.exit(-1)

        path_pictures = "images/changing_bw_3_byte_{}_{}{}/".format(height, width, path_suffix)
    
        orig_file_path = "images/orig_image_{}_{}.png".format(height, width)
        if same_image and os.path.exists(orig_file_path):
            pix_img = np.array(Image.open(orig_file_path))
        else:
            pix_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            Image.fromarray(pix_img).save(orig_file_path)

    pix_bws = get_pix_bws_from_pix_img(pix_img)
   
    if os.path.exists(path_pictures):
        os.system("rm -rf {}*".format(path_pictures))
    else:
        os.makedirs(path_pictures)
    
    path_animations = "images/animations/"
    path_movies = "images/movies/"
    if not os.path.exists(path_animations):
        os.makedirs(path_animations)
    if not os.path.exists(path_movies):
        os.makedirs(path_movies)
    

    def combine_1_byte_neighbours(pix_bws):
        def combine_1_bit_neighbours(pix_bws):
            pix = np.zeros(pix_bws[0].shape, dtype=np.uint8)
            for i, pix_bw in enumerate(pix_bws):
                pix += pix_bw<<i
            return pix

        pix_bw_channels = [combine_1_bit_neighbours(pix_bws[8*i:8*(i+1)]) for i in range(0, 3)]
        pix = np.zeros(pix_bw_channels[0].shape+(3, ), dtype=np.uint8)
        for i, pix_bw_c in enumerate(pix_bw_channels):
            pix[:, :, i] = pix_bw_c
        return pix

    pix_combine = combine_1_byte_neighbours(pix_bws)
    pix_combines = [pix_combine]

    bit_neighbor_manipulation = BitNeighborManipulation(ft=ft, with_frame=with_frame, path_lambda_functions=path_lambda_functions)

    # so long there are white pixels, repeat the elimination_process!
    # TODO: or add an termination point too!
    it = 1
    # repeat anything until it is complete blank / black / 0
    while np.sum([np.sum(pix_bw == 1) for pix_bw in pix_bws]) > 0 and (it < max_iterations if max_iterations > 0 else True):
        if it%10 == 0:
            print("it: {}".format(it))
        
        pix_bws = bit_neighbor_manipulation.apply_neighbor_logic(pix_bws)

        pix_combine = combine_1_byte_neighbours(pix_bws)
        pix_combines.append(pix_combine)
        it += 1

    # now take each image and interpolate between each image e.g. 10 samples
    def get_pix_between(pix_1, pix_2, alpha=0.5):
        return (pix_1.astype(np.float)*alpha+pix_2.astype(np.float)*(1.-alpha)).astype(np.uint8)
    
    arr_pix_combines = np.array(pix_combines)
    path_template = path_pictures+"rnd_{}_{}_bw_i_{{:03}}_{{:02}}.png".format(height, width)
    for i, (pix_1, pix_2) in enumerate(zip(pix_combines[:-1], pix_combines[1:])):
        Image.fromarray(pix_1).save(path_template.format(i, 0))
        for j in range(1, amount_combines):
            print("i: {}, j: {}".format(i, j))
            Image.fromarray(get_pix_between(pix_1, pix_2, float(amount_combines-j)/amount_combines)).save(path_template.format(i, j))

    Image.fromarray(arr_pix_combines[-1]).save(path_template.format(arr_pix_combines.shape[0]-1, 0))

    os.chdir("./{}".format(path_pictures))

    if height_resize != None and width_resize != None:
     if img_type == "random":
        for root_dir, dirs, files in os.walk("."):
            if not root_dir == ".":
                continue

            for file_name in files:
                if not ".png" in file_name or file_name == "orig_image.png":
                    print("continue: file_name: {}".format(file_name))
                    continue
                print("Resize, convert and reduce quality for file: '{}'".format(file_name))
                os.system("convert {} -filter Point -resize {}x{} +antialias {}".format(file_name, height_resize, width_resize, file_name))
    
    for root_dir, dirs, files in os.walk("."):
        if not root_dir == ".":
            continue

        arr = np.sort(np.array(files))
        file_num = 0
        for file_name in arr:
            if not ".png" in file_name or file_name == "orig_image.png":
                continue

            if file_num == 0:
                for _ in range(0, num_copies_first_image):
                    os.system("cp {} pic_{:04d}.png".format(file_name, file_num))
                    file_num += 1
            os.system("mv {} pic_{:04d}.png".format(file_name, file_num))
            file_num += 1

    random_64_bit_num = get_random_64_bit_number(4)
    suffix_temp = "_{}_{{}}_{{}}_{}".format(img_type, random_64_bit_num)
    suffix = suffix_temp.format(height, width)

    # suffix = "_{}_{}_{}_{}_{}".format(img_type, height, width, (lambda x: "-".join(list(map(str, x.bit_operators_idx[:x.max_bit_operators]))))(bit_neighbor_manipulation), random_64_bit_num)
    print("Create an animation (gif) with png's and suffix '{}'!".format(suffix))
    os.system("convert -delay {} -loop 0 *.png ../../{}animated{}.gif".format(gif_delay, path_animations, suffix))
    print("Create an animation (mp4) with png's and suffix '{}'!".format(suffix))
    os.system("ffmpeg -r {} -i pic_%04d.png -vcodec mpeg4 -y ../../{}movie{}.mp4".format(fps_movie, path_movies, suffix))

    os.chdir(prev_folder)
    if resize_params != None:
        os.chdir(path_animations)

        new_height = height-resize_params[0]-resize_params[1]
        new_width = width-resize_params[2]-resize_params[3]
        
        orig_animated_file_name = "animated{}.gif".format(suffix_temp.format(height, width))
        modif_animated_file_name = "animated{}.gif".format(suffix_temp.format(new_height, new_width))
        print("Now crop the image! (only when needed!)")
        os.system("convert {} -coalesce -repage 0x0 -crop {}x{}+{}+{} +repage {}".format(
            orig_animated_file_name,
            # new_width, new_height, resize_params[2], resize_params[0],
            new_width, new_height, resize_params[3], resize_params[1],
            modif_animated_file_name))


if __name__ == "__main__":
    # height = 64
    height = 128
    # height = 256
    # height = 512
    width = height

    height_resize = height*3
    width_resize = width*3

    # create_1_bit_neighbour_pictures(height, width)
    # create_1_byte_neighbour_pictures(height, width)
    # create_3_byte_neighbour_pictures("random", (height, width, False))

    # create_3_byte_neighbour_pictures("random", height=height, width=width, same_image=False, with_frame=True)
    # TODO: make the system call multithreaded!

    # create_from_image_neighbour_pictures("images/fall-autumn-red-season.jpg")
    # ## convert fall-autumn-red-season.jpg -resize 320x213 fall-autumn-red-season_resized.jpg
    
    max_iterations = 45
    resize_params = None
    ft = 3
    num_copies_first_image=4
    amount_combines=2
    gif_delay=5
    fps_movie=20

    # with open("lambda_functions/resize_params_2.pkl", "rb") as fout:
    #     dm = dill.load(fout)
    # resize_params = dm.resize_params
    # max_iterations = dm.max_iterations
    # print("resize_params: {}".format(resize_params))
    # print("max_iterations: {}".format(max_iterations))

    folder_suffix = ""
    argv = sys.argv
    if len(argv) > 1:
        folder_suffix = argv[1]

    create_3_byte_neighbour_pictures("random",
                                     height=height,
                                     width=width,
                                     same_image=True,
                                     height_resize=height_resize,
                                     width_resize=width_resize,
    
    # create_3_byte_neighbour_pictures("picture",
    #                                  image_path="images/fall-autumn-red-season_resized.jpg",
    #                                  resize_params=resize_params,

                                     with_frame=True,
                                     path_lambda_functions="lambda_functions/lambdas_5.txt",
                                     max_iterations=max_iterations,
                                     ft=ft,
                                     num_copies_first_image=num_copies_first_image,
                                     amount_combines=amount_combines,
                                     gif_delay=gif_delay,
                                     fps_movie=fps_movie,
                                     folder_suffix=folder_suffix)
