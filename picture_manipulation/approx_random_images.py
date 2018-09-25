#! /usr/bin/python3.6

import os
import pdb
import sys

import numpy as np

from PIL import Image

def apply_neighbour_logic(pix_bw, choosen_algo=0):
    # pos = np.where(pix_bw == 1)
    # print("len(pos[0]): {}".format(len(pos[0])))

    height, width = pix_bw.shape[:2]
    # pix_blank = np.zeros((height, width), dtype=np.uint8)
    # pix_u = pix_blank.copy()
    # pix_d = pix_blank.copy()
    # pix_l = pix_blank.copy()
    # pix_r = pix_blank.copy()

    # pix_ul = pix_blank.copy()
    # pix_ur = pix_blank.copy()
    # pix_dl = pix_blank.copy()
    # pix_dr = pix_blank.copy()
    
    # pix_uu = pix_blank.copy()
    # pix_dd = pix_blank.copy()
    # pix_ll = pix_blank.copy()
    # pix_rr = pix_blank.copy()

    # get_pos_u = lambda pos: (lambda x: (pos[0][x]-1, pos[1][x]))(pos[0]>0)
    # get_pos_d = lambda pos: (lambda x: (pos[0][x]+1, pos[1][x]))(pos[0]<(height-1))
    # get_pos_l = lambda pos: (lambda x: (pos[0][x], pos[1][x]-1))(pos[1]>0)
    # get_pos_r = lambda pos: (lambda x: (pos[0][x], pos[1][x]+1))(pos[1]<(width-1))

    # pos_u = get_pos_u(pos)
    # pos_d = get_pos_d(pos)
    # pos_l = get_pos_l(pos)
    # pos_r = get_pos_r(pos)
    
    # pos_ul = get_pos_l(pos_u)
    # pos_ur = get_pos_r(pos_u)
    # pos_dl = get_pos_l(pos_d)
    # pos_dr = get_pos_r(pos_d)

    # pos_uu = get_pos_u(pos_u)
    # pos_dd = get_pos_d(pos_d)
    # pos_ll = get_pos_l(pos_l)
    # pos_rr = get_pos_r(pos_r)

    # pix_u[pos_u] = 1
    # pix_d[pos_d] = 1
    # pix_l[pos_l] = 1
    # pix_r[pos_r] = 1

    # pix_ul[pos_ul] = 1
    # pix_ur[pos_ur] = 1
    # pix_dl[pos_dl] = 1
    # pix_dr[pos_dr] = 1

    # pix_uu[pos_uu] = 1
    # pix_dd[pos_dd] = 1
    # pix_ll[pos_ll] = 1
    # pix_rr[pos_rr] = 1

    zero_row = np.zeros((width, ), dtype=np.uint8)
    zero_col = np.zeros((height, 1), dtype=np.uint8)

    move_arr_u = lambda pix_bw: np.vstack((pix_bw[1:], zero_row))
    move_arr_d = lambda pix_bw: np.vstack((zero_row, pix_bw[:-1]))
    move_arr_l = lambda pix_bw: np.hstack((pix_bw[:, 1:], zero_col))
    move_arr_r = lambda pix_bw: np.hstack((zero_col, pix_bw[:, :-1]))

    # pix_u = move_arr_u(pix_bw)
    # pix_d = move_arr_d(pix_bw)
    # pix_l = move_arr_l(pix_bw)
    # pix_r = move_arr_r(pix_bw)
    
    # pix_ul = move_arr_l(pix_u)
    # pix_ur = move_arr_r(pix_u)
    # pix_dl = move_arr_l(pix_d)
    # pix_dr = move_arr_r(pix_d)

    # pix_uu = move_arr_u(pix_u)
    # pix_dd = move_arr_d(pix_d)
    # pix_ll = move_arr_l(pix_l)
    # pix_rr = move_arr_r(pix_r)

    pixs = np.zeros((5, 5, height, width), dtype=np.uint8)
    pixs[2, 2] = pix_bw
    
    for i in range(2, 0, -1):
        pixs[2, i-1] = move_arr_l(pixs[2, i])
    for i in range(2, 4):
        pixs[2, i+1] = move_arr_r(pixs[2, i])
    for i in range(2, 0, -1):
        pixs[i-1, 2] = move_arr_u(pixs[i, 2])
    for i in range(2, 4):
        pixs[i+1, 2] = move_arr_d(pixs[i, 2])

    for j in list(range(0, 2))+list(range(3, 5)):
        for i in range(2, 0, -1):
            pixs[j, i-1] = move_arr_l(pixs[j, i])
        for i in range(2, 4):
            pixs[j, i+1] = move_arr_r(pixs[j, i])

    pdb.set_trace()

    # print("np.sum(pix_u!=pix_u_2): {}".format(np.sum(pix_u!=pix_u_2)))
    # print("np.sum(pix_d!=pix_d_2): {}".format(np.sum(pix_d!=pix_d_2)))
    # print("np.sum(pix_l!=pix_l_2): {}".format(np.sum(pix_l!=pix_l_2)))
    # print("np.sum(pix_r!=pix_r_2): {}".format(np.sum(pix_r!=pix_r_2)))
    
    # print("np.sum(pix_ul!=pix_ul_2): {}".format(np.sum(pix_ul!=pix_ul_2)))
    # print("np.sum(pix_ur!=pix_ur_2): {}".format(np.sum(pix_ur!=pix_ur_2)))
    # print("np.sum(pix_dl!=pix_dl_2): {}".format(np.sum(pix_dl!=pix_dl_2)))
    # print("np.sum(pix_dr!=pix_dr_2): {}".format(np.sum(pix_dr!=pix_dr_2)))

    # print("np.sum(pix_uu!=pix_uu_2): {}".format(np.sum(pix_uu!=pix_uu_2)))
    # print("np.sum(pix_dd!=pix_dd_2): {}".format(np.sum(pix_dd!=pix_dd_2)))
    # print("np.sum(pix_ll!=pix_ll_2): {}".format(np.sum(pix_ll!=pix_ll_2)))
    # print("np.sum(pix_rr!=pix_rr_2): {}".format(np.sum(pix_rr!=pix_rr_2)))
    # sys.exit(2)

    idx_algo = [2, 3, 4, 5]
    fs = [
        lambda: \
        ((pix_u&pix_l|pix_d&pix_r)&(pix_bw == 1))^
        ((pix_u&pix_d|pix_l&pix_r)&(pix_bw == 0))|
        ((pix_u&pix_d&pix_l&pix_r)|(pix_bw == 0)&(pix_bw == 1)),
        
        lambda: \
        ((pix_u&pix_d|pix_l&pix_r)&(pix_bw == 0))^
        ((pix_ul&pix_ur|pix_dl&pix_dr)&(pix_bw == 1)),

        lambda:
        (((pix_uu&pix_u)^(pix_dd&pix_d))&(pix_bw==0))|
        (((pix_ll&pix_l)|(pix_rr&pix_r))&(pix_bw==1)),
        lambda:
        (((pix_uu&pix_u)&(pix_dd&pix_d))&(pix_bw==1))|
        (((pix_ll&pix_l)^(pix_rr&pix_r))&(pix_bw==0)),
        
        lambda:
        ((pix_uu&pix_dd)&(pix_bw==0))|
        ((pix_ll&pix_rr)&(pix_bw==1)),
        lambda:
        ((pix_uu&pix_dd)&(pix_bw==1))^
        ((pix_ll&pix_rr)&(pix_bw==0)),

        lambda: \
        ((pix_ul&pix_u&pix_ur&pix_d)&(pix_bw == 0))^
        ((pix_l&pix_r&pix_dl&pix_dr)&(pix_bw == 1)),
        lambda: \
        ((pix_dl&pix_d&pix_dr&pix_u)&(pix_bw == 0))^
        ((pix_l&pix_r&pix_ul&pix_ur)&(pix_bw == 1)),
        lambda: \
        ((pix_ul&pix_l&pix_dl&pix_r)&(pix_bw == 0))^
        ((pix_u&pix_d&pix_ur&pix_dr)&(pix_bw == 1)),
        lambda: \
        ((pix_ur&pix_r&pix_dr&pix_l)&(pix_bw == 0))^
        ((pix_u&pix_d&pix_ul&pix_dl)&(pix_bw == 1)),

        lambda: \
        ((pix_u&pix_d&pix_l&pix_r)&(pix_bw == 0))^
        ((pix_ul&pix_ur&pix_dl&pix_dr)&(pix_bw == 1)),
        lambda: \
        ((pix_u&pix_d&pix_l&pix_r)&(pix_bw == 1))^
        ((pix_ul&pix_ur&pix_dl&pix_dr)&(pix_bw == 0)),
        lambda: \
        ((pix_u&pix_d|pix_l&pix_r)&(pix_bw == 0))|
        ((pix_ul&pix_ur|pix_dl&pix_dr)&(pix_bw == 1)),
        
        lambda: \
        (pix_u&pix_ur&pix_r&pix_dr)|
        (pix_ur&pix_r&pix_dr&pix_d)|
        (pix_r&pix_dr&pix_d&pix_dl)|
        (pix_dr&pix_d&pix_dl&pix_l)|
        (pix_d&pix_dl&pix_l&pix_ul)|
        (pix_dl&pix_l&pix_ul&pix_u)|
        (pix_l&pix_ul&pix_u&pix_ur)|
        (pix_ul&pix_u&pix_ur&pix_r),
        lambda: \
        (pix_u&pix_ur&pix_r&pix_dr&pix_d)|
        (pix_ur&pix_r&pix_dr&pix_d&pix_dl)|
        (pix_r&pix_dr&pix_d&pix_dl&pix_l)|
        (pix_dr&pix_d&pix_dl&pix_l&pix_ul)|
        (pix_d&pix_dl&pix_l&pix_ul&pix_u)|
        (pix_dl&pix_l&pix_ul&pix_u&pix_ur)|
        (pix_l&pix_ul&pix_u&pix_ur&pix_r)|
        (pix_ul&pix_u&pix_ur&pix_r&pix_dr),

        lambda: \
        (pix_u&pix_ur&pix_r)|
        (pix_ur&pix_r&pix_dr)|
        (pix_r&pix_dr&pix_d)|
        (pix_dr&pix_d&pix_dl)|
        (pix_d&pix_dl&pix_l)|
        (pix_dl&pix_l&pix_ul)|
        (pix_l&pix_ul&pix_u)|
        (pix_ul&pix_u&pix_ur),
        
        lambda: \
        ((pix_u&pix_r)|(pix_l&pix_d))&((pix_ul&pix_ur)|(pix_dl&pix_dr)),
        
        lambda: \
        ((pix_u&pix_d)|(pix_l&pix_r))^((pix_ul&pix_dr)|(pix_dl&pix_ur)),
        lambda: \
        ((pix_u&pix_r)|(pix_l&pix_d))^((pix_ul&pix_ur)|(pix_dl&pix_dr)),
        
        lambda: \
        ((pix_u&pix_d|pix_l&pix_r)&(pix_bw == 0))|
        ((pix_ul&pix_ur|pix_dl&pix_dr)&(pix_bw == 1)),
        lambda: \
        ((pix_u&pix_d|pix_l&pix_r)&(pix_bw == 1))|
        ((pix_ul&pix_ur|pix_dl&pix_dr)&(pix_bw == 0)),
        
        lambda: \
        ((pix_ul&pix_u&pix_ur|pix_dl&pix_d&pix_dr)&(pix_bw == 0))|
        ((pix_ul&pix_l&pix_dl|pix_ur&pix_r&pix_dr)&(pix_bw == 1)),
        lambda: \
        ((pix_ul&pix_u&pix_ur|pix_dl&pix_d&pix_dr)&(pix_bw == 1))|
        ((pix_ul&pix_l&pix_dl|pix_ur&pix_r&pix_dr)&(pix_bw == 0)),
        
        lambda: \
        ((pix_ul&pix_u&pix_ur|pix_ul&pix_l&pix_dl)&(pix_bw == 0))|
        ((pix_dl&pix_d&pix_dr|pix_ur&pix_r&pix_dr)&(pix_bw == 1)),
        lambda: \
        ((pix_ul&pix_u&pix_ur|pix_ul&pix_l&pix_dl)&(pix_bw == 1))|
        ((pix_dl&pix_d&pix_dr|pix_ur&pix_r&pix_dr)&(pix_bw == 0)),
        
        lambda: \
        ((pix_ul&pix_u&pix_ur|pix_ur&pix_r&pix_dr)&(pix_bw == 0))|
        ((pix_dl&pix_d&pix_dr|pix_ul&pix_l&pix_dl)&(pix_bw == 1)),
        lambda: \
        ((pix_ul&pix_u&pix_ur|pix_ur&pix_r&pix_dr)&(pix_bw == 1))|
        ((pix_dl&pix_d&pix_dr|pix_ul&pix_l&pix_dl)&(pix_bw == 0))
        ]
    # pix_bw = ((((pix_u&pix_l|pix_d&pix_r)&(pix_bw == 1))^
    #            ((pix_u&pix_d|pix_l&pix_r)&(pix_bw == 0))|
    #            ((pix_u&pix_d&pix_l&pix_r)|(pix_bw == 0)&(pix_bw == 1)))*1).astype(np.uint8)
    pix_bw = fs[idx_algo[choosen_algo]]().astype(np.uint8)

    # if choosen_algo == 0:
    #     pix_bw = (((pix_u&pix_d)^(pix_l&pix_r))&
    #               ((pix_u&pix_l)^(pix_d&pix_r))|
    #               ((pix_u&pix_d&pix_l)^(pix_r))&
    #               ((pix_u&pix_r&pix_l)^(pix_d))|
    #               ((pix_u&(pix_l==0))&(pix_bw==0))&
    #               ((pix_d&(pix_r==0))&(pix_bw==1))|
    #               ((pix_u&(pix_l==1))&(pix_bw==1))&
    #               ((pix_d&(pix_r==1))&(pix_bw==0))).astype(np.uint8)
    # elif choosen_algo == 1:
    #     pix_bw = ((pix_ur&pix_dl)^
    #               (pix_ul&pix_dr)|
    #               (pix_u&pix_d)^
    #               (pix_l&pix_r)).astype(np.uint8)
    # elif choosen_algo == 2:
    #     pix_bw = ((pix_ur&pix_ul&pix_dr&pix_dl)|
    #               (pix_u&pix_d&pix_l&pix_r)).astype(np.uint8)
    # elif choosen_algo == 3:
    #     pix_bw = ((pix_u+pix_d+pix_l+pix_r+pix_ur+pix_ul+pix_dr+pix_dl+pix_bw) > 5).astype(np.uint8)
    # elif choosen_algo == 4:
    #     pix_bw = ((((pix_u&pix_l|pix_d&pix_r)&(pix_bw == 1))^((pix_u&pix_d|pix_l&pix_r)&(pix_bw == 0)))*1).astype(np.uint8)
    # else:
    #     pix_bw = (((pix_u&pix_l|pix_d&pix_r)&(pix_bw == 1))*1).astype(np.uint8)

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
    
    # prev_folder = os.getcwd()
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
            pix_bws[i] = apply_neighbour_logic(pix_bws[i], choosen_algo=i%2)

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
            os.system("convert {} -filter Point -resize 256x256 +antialias {}".format(file_name, file_name))
            # os.system("mogrify -format jpg {}".format(file_name))
            # file_name_jpg = file_name.replace(".png", ".jpg")
            # os.system("convert {} -quality 20% {}".format(file_name_jpg, file_name_jpg))
    
    print("Create an animation with png's!")
    os.system("convert -delay 5 -loop 0 *.png animated_png.gif")
    # print("Create an animation with jpg's!")
    # os.system("convert -delay 2 -loop 0 *.jpg animated_jpg.gif")

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
    height = 64
    # height = 128
    # height = 256
    # height = 512
    width = height

    # create_1_bit_neighbour_pictures(height, width)
    # create_1_byte_neighbour_pictures(height, width)
    # create_3_byte_neighbour_pictures(height, width)    
    # create_from_image_neighbour_pictures("images/fall-autumn-red-season.jpg")
    ## convert fall-autumn-red-season.jpg -resize 320x213 fall-autumn-red-season_resized.jpg
    create_from_image_neighbour_pictures("images/fall-autumn-red-season_resized.jpg")
