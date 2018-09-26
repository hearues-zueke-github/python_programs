#! /usr/bin/python3.6

import os
import pdb
import string
import sys

import numpy as np

from PIL import Image

def get_random_64_bit_number(n):
    s = string.ascii_lowercase+string.ascii_uppercase+string.digits+"-_"
    arr = np.array(list(s))
    return "".join(np.random.choice(arr, (n, )).tolist())

# TODO: create an object out of these apply able functions
def apply_neighbour_logic(pix_bw, choosen_algo=0):
    ft = 3 # frame_thickness
    # add the frame to the image with itself
    # e.g. the left 2 cols add to the right side, and vice versa for all other sides
    pix_bw = (lambda x: np.hstack((x[:, -ft:], x, x[:, :ft])))(np.vstack((pix_bw[-ft:], pix_bw, pix_bw[:ft])).copy())
    
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
    for i in range(ft, 4):
        pixs[i+1, ft] = move_arr_d(pixs[i, ft])

    # then set all x pixs (except the center ones, they are already set)
    for j in range(0, 5):
        for i in range(ft, 0, -1):
            pixs[j, i-1] = move_arr_l(pixs[j, i])
        for i in range(ft, 4):
            pixs[j, i+1] = move_arr_r(pixs[j, i])

    # now define for each pix a variable name
    # e.g. p_urr = pixs[1, 4] for ft == 2
    # or   p_ddll = pixs[4, 0] also for ft == 2
    # or   p_ulll = pixs[2, 0] for ft == 3, and so on...
    variables = ["p"]
    p = pixs[ft, ft]
    for y in range(0, ft*2+1):
        for x in range(0, ft*2+1):
            if y == ft and x == ft:
                continue
            var_name = "p_"+("u"*(ft-y) if y < ft else "d"*(y-ft))+("l"*(ft-x) if x < ft else "r"*(x-ft))
            variables.append(var_name)
            exec("globals()['{}'] = pixs[{}, {}]".format(var_name, y, x))
    
    pdb.set_trace()

    idx_algo = [0, 1, 3, 4, 2, 11, 7, 9]
    fs = [
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
        ((p_u&p_d&p_l&p_r)),
        
        lambda: \
        ((p_u&p_d|p_l&p_r)&(p==0))^
        ((p_ul&p_ur|p_dl&p_dr)&(p==1)),

        lambda:
        (((p_uu&p_u)^(p_dd&p_d))&(p==0))|
        (((p_ll&p_l)|(p_rr&p_r))&(p==1)),
        lambda:
        (((p_uu&p_u)&(p_dd&p_d))&(p==1))|
        (((p_ll&p_l)^(p_rr&p_r))&(p==0)),
        
        lambda:
        ((p_uu&p_dd)&(p==0))|
        ((p_ll&p_rr)&(p==1)),
        lambda:
        ((p_uu&p_dd)&(p==1))^
        ((p_ll&p_rr)&(p==0)),

        lambda: \
        ((p_ul&p_u&p_ur&p_d)&(p==0))^
        ((p_l&p_r&p_dl&p_dr)&(p==1)),
        lambda: \
        ((p_dl&p_d&p_dr&p_u)&(p==0))^
        ((p_l&p_r&p_ul&p_ur)&(p==1)),
        lambda: \
        ((p_ul&p_l&p_dl&p_r)&(p==0))^
        ((p_u&p_d&p_ur&p_dr)&(p==1)),
        lambda: \
        ((p_ur&p_r&p_dr&p_l)&(p==0))^
        ((p_u&p_d&p_ul&p_dl)&(p==1)),

        lambda: \
        ((p_u&p_d&p_l&p_r)&(p==0))^
        ((p_ul&p_ur&p_dl&p_dr)&(p==1)),
        lambda: \
        ((p_u&p_d&p_l&p_r)&(p==1))^
        ((p_ul&p_ur&p_dl&p_dr)&(p==0)),
        lambda: \
        ((p_u&p_d|p_l&p_r)&(p==0))|
        ((p_ul&p_ur|p_dl&p_dr)&(p==1)),
        
        lambda: \
        (p_u&p_ur&p_r&p_dr)|
        (p_ur&p_r&p_dr&p_d)|
        (p_r&p_dr&p_d&p_dl)|
        (p_dr&p_d&p_dl&p_l)|
        (p_d&p_dl&p_l&p_ul)|
        (p_dl&p_l&p_ul&p_u)|
        (p_l&p_ul&p_u&p_ur)|
        (p_ul&p_u&p_ur&p_r),
        lambda: \
        (p_u&p_ur&p_r&p_dr&p_d)|
        (p_ur&p_r&p_dr&p_d&p_dl)|
        (p_r&p_dr&p_d&p_dl&p_l)|
        (p_dr&p_d&p_dl&p_l&p_ul)|
        (p_d&p_dl&p_l&p_ul&p_u)|
        (p_dl&p_l&p_ul&p_u&p_ur)|
        (p_l&p_ul&p_u&p_ur&p_r)|
        (p_ul&p_u&p_ur&p_r&p_dr),

        lambda: \
        (p_u&p_ur&p_r)|
        (p_ur&p_r&p_dr)|
        (p_r&p_dr&p_d)|
        (p_dr&p_d&p_dl)|
        (p_d&p_dl&p_l)|
        (p_dl&p_l&p_ul)|
        (p_l&p_ul&p_u)|
        (p_ul&p_u&p_ur),
        
        lambda: \
        ((p_u&p_r)|(p_l&p_d))&((p_ul&p_ur)|(p_dl&p_dr)),
        
        lambda: \
        ((p_u&p_d)|(p_l&p_r))^((p_ul&p_dr)|(p_dl&p_ur)),
        lambda: \
        ((p_u&p_r)|(p_l&p_d))^((p_ul&p_ur)|(p_dl&p_dr)),
        
        lambda: \
        ((p_u&p_d|p_l&p_r)&(p==0))|
        ((p_ul&p_ur|p_dl&p_dr)&(p==1)),
        lambda: \
        ((p_u&p_d|p_l&p_r)&(p==1))|
        ((p_ul&p_ur|p_dl&p_dr)&(p==0)),
        
        lambda: \
        ((p_ul&p_u&p_ur|p_dl&p_d&p_dr)&(p==0))|
        ((p_ul&p_l&p_dl|p_ur&p_r&p_dr)&(p==1)),
        lambda: \
        ((p_ul&p_u&p_ur|p_dl&p_d&p_dr)&(p==1))|
        ((p_ul&p_l&p_dl|p_ur&p_r&p_dr)&(p==0)),
        
        lambda: \
        ((p_ul&p_u&p_ur|p_ul&p_l&p_dl)&(p==0))|
        ((p_dl&p_d&p_dr|p_ur&p_r&p_dr)&(p==1)),
        lambda: \
        ((p_ul&p_u&p_ur|p_ul&p_l&p_dl)&(p==1))|
        ((p_dl&p_d&p_dr|p_ur&p_r&p_dr)&(p==0)),
        
        lambda: \
        ((p_ul&p_u&p_ur|p_ur&p_r&p_dr)&(p==0))|
        ((p_dl&p_d&p_dr|p_ul&p_l&p_dl)&(p==1)),
        lambda: \
        ((p_ul&p_u&p_ur|p_ur&p_r&p_dr)&(p==1))|
        ((p_dl&p_d&p_dr|p_ul&p_l&p_dl)&(p==0))
        ]

    pix_bw_1 = fs[idx_algo[choosen_algo]]()
    pix_bw_2 = fs[idx_algo[choosen_algo+1]]()
    pix_bw_3 = fs[idx_algo[choosen_algo+2]]()
    pix_bw = (pix_bw_1^pix_bw_2^pix_bw_3).astype(np.uint8)

    # remove the frame from the image again
    return pix_bw[2:-2, 2:-2]
    # return pix_bw

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
            pix_bws[i] = apply_neighbour_logic(pix_bws[i])

        pix_combine = combine_1_bit_neighbours(pix_bws)
        Image.fromarray(pix_combine).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))
        it += 1

def create_3_byte_neighbour_pictures(img_type, f_args):
    if img_type == "picture":
        image_path = f_args[0]

        if not os.path.exists(image_path):
            print("Path to image '{}' does not exists!".format(image_path))
            return -1

        img = Image.open(image_path)
        pix_img = np.array(img)
        height, width = pix_img.shape[:2]

        path_pictures = "images/changing_image_{}_{}/".format(height, width)
        
        pix_bws = [(pix_c>>j)&0x1 for pix_c in [pix_img[:, :, i] for i in range(0, 3)] for j in range(0, 8)]

    elif img_type == "random":
        height = f_args[0]
        width = f_args[1]

        path_pictures = "images/changing_bw_3_byte_{}_{}/".format(height, width)
    
        get_pix_bw = lambda: np.random.randint(0, 2, (height, width), dtype=np.uint8)
        pix_bws = [get_pix_bw() for _ in range(0, 24)]
    
    if os.path.exists(path_pictures):
        os.system("rm -rf {}".format(path_pictures))
    if not os.path.exists(path_pictures):
        os.makedirs(path_pictures)
    
    path_animations = "images/animations/"
    path_movies = "images/movies/"
    if not os.path.exists(path_animations):
        os.makedirs(path_animations)
    if not os.path.exists(path_movies):
        os.makedirs(path_movies)
    
    prev_folder = os.getcwd()

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

    # so long there are white pixels, repeat the elimination_process!
    it = 1
    # repeat anything until it is complete blank / black / 0
    while np.sum([np.sum(pix_bw == 1) for pix_bw in pix_bws]) > 0:
        if it%10 == 0:
            print("it: {}".format(it))
        
        for i in range(0, 24):
            pix_bws[i] = apply_neighbour_logic(pix_bws[i], choosen_algo=it%4)

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
        amount_combines = 5
        for j in range(1, amount_combines):
            print("i: {}, j: {}".format(i, j))
            Image.fromarray(get_pix_between(pix_1, pix_2, float(amount_combines-j)/amount_combines)).save(path_template.format(i, j))

    Image.fromarray(arr_pix_combines[-1]).save(path_template.format(arr_pix_combines.shape[0]-1, 0))

    os.chdir("./{}".format(path_pictures))

    if img_type == "random":
        for root_dir, dirs, files in os.walk("."):
            if not root_dir == ".":
                continue

            for file_name in files:
                if not ".png" in file_name:
                    print("continue: file_name: {}".format(file_name))
                    continue
                print("Resize, convert and reduce quality for file: '{}'".format(file_name))
                os.system("convert {} -filter Point -resize 256x256 +antialias {}".format(file_name, file_name))
    
    for root_dir, dirs, files in os.walk("."):
        if not root_dir == ".":
            continue

        arr = np.sort(np.array(files))
        file_num = 0
        for file_name in arr:
            if file_num == 0:
                for _ in range(0, 3):
                    os.system("cp {} pic_{:04d}.png".format(file_name, file_num))
                    file_num += 1
            os.system("mv {} pic_{:04d}.png".format(file_name, file_num))
            file_num += 1

    random_64_bit_num = get_random_64_bit_number(4)
    suffix = "_{}_{}_{}_{}".format(img_type, height, width, random_64_bit_num)
    print("Create an animation (gif) with png's and suffix '{}'!".format(suffix))
    os.system("convert -delay 5 -loop 0 *.png ../../{}animated{}.gif".format(path_animations, suffix))
    print("Create an animation (mp4) with png's and suffix '{}'!".format(suffix))
    os.system("ffmpeg -r 20 -i pic_%04d.png -vcodec mpeg4 -y ../../{}movie{}.mp4".format(path_movies, suffix))

    os.chdir(prev_folder)

if __name__ == "__main__":
    # height = 64
    height = 128
    # height = 256
    # height = 512
    width = height

    # create_1_bit_neighbour_pictures(height, width)
    # create_1_byte_neighbour_pictures(height, width)
    create_3_byte_neighbour_pictures("random", (height, width))
    # create_from_image_neighbour_pictures("images/fall-autumn-red-season.jpg")
    # ## convert fall-autumn-red-season.jpg -resize 320x213 fall-autumn-red-season_resized.jpg
    create_3_byte_neighbour_pictures("picture", ("images/fall-autumn-red-season_resized.jpg", ))
