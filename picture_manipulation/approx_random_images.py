#! /usr/bin/python3.6

import os
import pdb
import sys

import numpy as np

from PIL import Image

def apply_neighbour_logic(pix_bw, choosen_algo=0):
    # # add the frame to the image with itself
    # # e.g. the left 2 cols add to the right side, and vice versa for all other sides
    # pix_bw = (lambda x: np.hstack((x[:, -2:], x, x[:, :2])))(np.vstack((pix_bw[-2:], pix_bw, pix_bw[:2])).copy())
    
    height, width = pix_bw.shape[:2]

    zero_row = np.zeros((width, ), dtype=np.uint8)
    zero_col = np.zeros((height, 1), dtype=np.uint8)

    move_arr_u = lambda pix_bw: np.vstack((pix_bw[1:], zero_row))
    move_arr_d = lambda pix_bw: np.vstack((zero_row, pix_bw[:-1]))
    move_arr_l = lambda pix_bw: np.hstack((pix_bw[:, 1:], zero_col))
    move_arr_r = lambda pix_bw: np.hstack((zero_col, pix_bw[:, :-1]))

    pixs = np.zeros((5, 5, height, width), dtype=np.uint8)
    pixs[2, 2] = pix_bw

    for i in range(2, 0, -1):
        pixs[i-1, 2] = move_arr_u(pixs[i, 2])
    for i in range(2, 4):
        pixs[i+1, 2] = move_arr_d(pixs[i, 2])

    for j in range(0, 5):
        for i in range(2, 0, -1):
            pixs[j, i-1] = move_arr_l(pixs[j, i])
        for i in range(2, 4):
            pixs[j, i+1] = move_arr_r(pixs[j, i])

    var_map = {(2, 2): "pix",

               (1, 2): "p_u",
               (3, 2): "p_d",
               (2, 1): "p_l",
               (2, 3): "p_r",

               (1, 1): "p_ul",
               (1, 3): "p_ur",
               (3, 1): "p_dl",
               (3, 3): "p_dr",

               (0, 2): "p_uu",
               (4, 2): "p_dd",
               (2, 0): "p_ll",
               (2, 4): "p_rr",

               (0, 1): "p_uul",
               (0, 3): "p_uur",
               (4, 1): "p_ddl",
               (4, 3): "p_ddr",
               (1, 0): "p_ull",
               (1, 4): "p_urr",
               (3, 0): "p_dll",
               (3, 4): "p_drr",

               (0, 0): "p_uull",
               (0, 4): "p_uurr",
               (4, 0): "p_ddll",
               (4, 4): "p_ddrr"}

    for key, value in var_map.items():
        exec("globals()['{}'] = pixs[{}, {}]".format(value, key[0], key[1]))

    p = pix_bw
    idx_algo = [1, 3, 4, 2, 5]
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

    # # remove the frame from the image again
    # return pix_bw[2:-2, 2:-2]
    return pix_bw

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

def create_3_byte_neighbour_pictures(height, width):
    path_pictures = "images/changing_bw_3_byte_{}_{}/".format(height, width)
    
    if os.path.exists(path_pictures):
        os.system("rm -rf {}".format(path_pictures))
    if not os.path.exists(path_pictures):
        os.makedirs(path_pictures)
    
    prev_folder = os.getcwd()
    # os.chdir("./{}".format(path_pictures))
    # os.chdir(prev_folder)

    get_pix_bw = lambda: np.random.randint(0, 2, (height, width), dtype=np.uint8)
    pix_bws = [get_pix_bw() for _ in range(0, 24)]

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
    # Image.fromarray(pix_combine).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, 0))

    # so long there are white pixels, repeat the elimination_process!
    it = 1
    # repeat anything until it is complete blank / black / 0
    while np.sum([np.sum(pix_bw == 1) for pix_bw in pix_bws]) > 0:
        print("it: {}".format(it))
        
        for i in range(0, 24):
            pix_bws[i] = apply_neighbour_logic(pix_bws[i], choosen_algo=it%2)

        pix_combine = combine_1_byte_neighbours(pix_bws)
        pix_combines.append(pix_combine)
        # Image.fromarray(pix_combine).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))
        it += 1

    # now take each image and interpolate between each image e.g. 10 samples
    def get_pix_between(pix_1, pix_2, alpha=0.5):
        return (pix_1.astype(np.float)*alpha+pix_2.astype(np.float)*(1.-alpha)).astype(np.uint8)
    arr_pix_combines = np.array(pix_combines)
    path_template = path_pictures+"rnd_{}_{}_bw_i_{{:03}}_{{:02}}.png".format(height, width)
    for i, (pix_1, pix_2) in enumerate(zip(pix_combines[:-1], pix_combines[1:])):
        Image.fromarray(pix_1).save(path_template.format(i, 0))
        amount_combines = 1
        for j in range(1, amount_combines):
            print("i: {}, j: {}".format(i, j))
            Image.fromarray(get_pix_between(pix_1, pix_2, float(amount_combines-j)/amount_combines)).save(path_template.format(i, j))

    Image.fromarray(arr_pix_combines[-1]).save(path_template.format(arr_pix_combines.shape[0]-1, 0))

    # os.system("ls")
    os.chdir("./{}".format(path_pictures))

    for root_dir, dirs, files in os.walk("."):
        # print("root_dir: {}".format(root_dir))
        # print("files: {}".format(files))
        # print("dirs: {}".format(dirs))
        if not root_dir == ".":
            continue

        for file_name in files:
            if not ".png" in file_name:
                print("continue: file_name: {}".format(file_name))
                continue
            print("Resize, convert and reduce quality for file: '{}'".format(file_name))
            os.system("convert {} -filter Point -resize 256x256 +antialias {}".format(file_name, file_name))
            # os.system("mogrify -format jpg {}".format(file_name))
            # file_name_jpg = file_name.replace(".png", ".jpg")
            # os.system("convert {} -quality 20% {}".format(file_name_jpg, file_name_jpg))
    
    for root_dir, dirs, files in os.walk("."):
        if not root_dir == ".":
            continue

        arr = np.sort(np.array(files))
        for i, file_name in enumerate(arr):
            os.system("mv {} pic_{:04d}.png".format(file_name, i))

    # print("Create an animation (gif) with png's!")
    # os.system("convert -delay 5 -loop 0 *.png animated_png.gif")
    print("Create an animation (mp4) with png's!")
    os.system("ffmpeg -r 20 -i pic_%04d.png -vcodec mpeg4 -y movie.mp4")

    os.chdir(prev_folder)

def create_from_image_neighbour_pictures(image_path):
    if not os.path.exists(image_path):
        print("Path to image '{}' does not exists!".format(image_path))
        sys.exit(-1)

    img = Image.open(image_path)
    # img.show()
    pix_img = np.array(img)
    height, width = pix_img.shape[:2]
    # print("height: {}, width: {}".format(height, width))
    # print("height//2: {}, width//2: {}".format(height//2, width//2))
    # print("height//4: {}, width//4: {}".format(height//4, width//4))
    # sys.exit(0)

    path_pictures = "images/changing_image_{}_{}/".format(height, width)
    
    if os.path.exists(path_pictures):
        os.system("rm -rf {}".format(path_pictures))
    if not os.path.exists(path_pictures):
        os.makedirs(path_pictures)
    
    prev_folder = os.getcwd()
    # os.chdir("./{}".format(path_pictures))

    # get_pix_bw = lambda: np.random.randint(0, 2, (height, width), dtype=np.uint8)
    # pix_bws = [get_pix_bw() for _ in range(0, 24)]
    pix_bws = [(pix_c>>j)&0x1 for pix_c in [pix_img[:, :, i] for i in range(0, 3)] for j in range(0, 8)]

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
    # img_combine = Image.fromarray(pix_combine)
    # img_combine.show()
    # print("pix_img.shape: {}".format(pix_img.shape))
    # print("pix_combine.shape: {}".format(pix_combine.shape))
    # not_equal_bytes = np.sum(pix_img!=pix_combine)
    # print("not_equal_bytes: {}".format(not_equal_bytes))
    # Image.fromarray(pix_img).show()
    # sys.exit(0)
    pix_combines = [pix_combine]
    # Image.fromarray(pix_combine).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, 0))

    # so long there are white pixels, repeat the elimination_process!
    it = 1
    # repeat anything until it is complete blank / black / 0
    while np.sum([np.sum(pix_bw == 1) for pix_bw in pix_bws]) > 0:
        print("it: {}".format(it))
        
        for i in range(0, 24):
            pix_bws[i] = apply_neighbour_logic(pix_bws[i], choosen_algo=(it)%2)

        pix_combine = combine_1_byte_neighbours(pix_bws)
        pix_combines.append(pix_combine)
        # Image.fromarray(pix_combine).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))
        it += 1

    # now take each image and interpolate between each image e.g. 10 samples
    def get_pix_between(pix_1, pix_2, alpha=0.5):
        return (pix_1.astype(np.float)*alpha+pix_2.astype(np.float)*(1.-alpha)).astype(np.uint8)
    arr_pix_combines = np.array(pix_combines)
    path_template = path_pictures+"rnd_{}_{}_bw_i_{{:03}}_{{:02}}.png".format(height, width)
    for i, (pix_1, pix_2) in enumerate(zip(pix_combines[:-1], pix_combines[1:])):
        Image.fromarray(pix_1).save(path_template.format(i, 0))
        amount_combines = 2
        for j in range(1, amount_combines):
            print("i: {}, j: {}".format(i, j))
            Image.fromarray(get_pix_between(pix_1, pix_2, float(amount_combines-j)/amount_combines)).save(path_template.format(i, j))

    Image.fromarray(arr_pix_combines[-1]).save(path_template.format(arr_pix_combines.shape[0]-1, 0))

    # os.system("ls")
    os.chdir("./{}".format(path_pictures))

    for root_dir, dirs, files in os.walk("."):
        # print("root_dir: {}".format(root_dir))
        # print("files: {}".format(files))
        # print("dirs: {}".format(dirs))
        if not root_dir == ".":
            continue

        for file_name in files:
            if not ".png" in file_name:
                print("continue: file_name: {}".format(file_name))
                continue
            print("Resize, convert and reduce quality for file: '{}'".format(file_name))
            # os.system("convert {} -filter Point -resize 256x256 +antialias {}".format(file_name, file_name))
            # os.system("mogrify -format jpg {}".format(file_name))
            # file_name_jpg = file_name.replace(".png", ".jpg")
            # os.system("convert {} -quality 20% {}".format(file_name_jpg, file_name_jpg))
    
    
    # print("Create an animation with jpg's!")
    # os.system("convert -delay 2 -loop 0 *.jpg animated_jpg.gif")
    
    for root_dir, dirs, files in os.walk("."):
        if not root_dir == ".":
            continue

        arr = np.sort(np.array(files))
        for i, file_name in enumerate(arr):
            os.system("mv {} pic_{:04d}.png".format(file_name, i))

    # pdb.set_trace()

    # print("Create an animation (gif) with png's!")
    # os.system("convert -delay 5 -loop 0 *.png animated_png.gif")
    print("Create an animation (mp4) with png's!")
    os.system("ffmpeg -r 20 -i pic_%04d.png -vcodec mpeg4 -y movie.mp4")

    os.chdir(prev_folder)

if __name__ == "__main__":
    # height = 64
    height = 128
    # height = 256
    # height = 512
    width = height

    # create_1_bit_neighbour_pictures(height, width)
    # create_1_byte_neighbour_pictures(height, width)
    create_3_byte_neighbour_pictures(height, width)    
    # create_from_image_neighbour_pictures("images/fall-autumn-red-season.jpg")
    # ## convert fall-autumn-red-season.jpg -resize 320x213 fall-autumn-red-season_resized.jpg
    create_from_image_neighbour_pictures("images/fall-autumn-red-season_resized.jpg")
