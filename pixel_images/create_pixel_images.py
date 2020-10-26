#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import re
import string
import sys

import numpy as np
import scipy.stats as st
# import scipy as sp

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

from memory_tempfile import MemoryTempfile
tempfile = MemoryTempfile()

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = tempfile.gettempdir()+"/"

from config_file import FILE_NAME_JPG

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()
# sys.exit(0)

"""
pix ... grayscale image!
boder_amount ... how many pixels the border should be copied outside!
"""
arr_direction_colors = np.array([
    [0x00, 0x00, 0x00],
    [0xFF, 0x00, 0x00],
    [0x00, 0xFF, 0x00],
    [0x00, 0x00, 0xFF],
    [0xFF, 0xFF, 0x00],
    [0xFF, 0x00, 0xFF],
    [0x00, 0xFF, 0xFF],
    [0x80, 0x80, 0xFF],
    [0x80, 0xFF, 0x80],
], dtype=np.uint8)
def create_sobel_image_derivative(pix, border_amount, dir_path):
    pix_int = pix.astype(int)
    pix_int_border_h = np.hstack((pix_int[:, :1], )*border_amount+(pix_int, )+(pix_int[:, -1:], )*border_amount)
    pix_int_border = np.vstack((pix_int_border_h[:1, :], )*border_amount+(pix_int_border_h, )+(pix_int_border_h[-1:, :], )*border_amount)

    pix_float_border = pix_int_border.astype(np.float) / 255.

    print("pix.shape: {}".format(pix.shape))
    print("pix_int_border.shape: {}".format(pix_int_border.shape))

    arr_weight_part = np.array([[1, 2, 1]])

    for i in range(2, border_amount+1):
        arr_col = np.zeros((arr_weight_part.shape[0], 1), dtype=arr_weight_part.dtype)
        arr_weight_part_1 = np.hstack((arr_col, arr_weight_part, arr_col))
        arr_weight_part = np.vstack((arr_weight_part_1[:1, :]+1, arr_weight_part_1))

    arr_weight = arr_weight_part.T
    arr_weight_x = np.hstack((-np.flip(arr_weight, 1), np.zeros((arr_weight.shape[0], 1), dtype=arr_weight.dtype), arr_weight))
    arr_weight_y = np.flip(arr_weight_x.T, 0)

    print("arr_weight_x:\n{}".format(arr_weight_x))
    print("arr_weight_y:\n{}".format(arr_weight_y))

    h, w = pix.shape

    pb_sobel_x_float = np.zeros(pix_int.shape, dtype=np.float)
    # pb_sobel_x_int = pix_int.copy()
    for j, arr_row in enumerate(arr_weight_x, 0):
        for i, v in enumerate(arr_row, 0):
            pb_sobel_x_float += v*pix_float_border[j:j+h, i:i+w]
            # pb_sobel_x_int += v*pix_int_border[j:j+h, i:i+w]
    
    pb_sobel_y_float = np.zeros(pix_int.shape, dtype=np.float)
    # pb_sobel_y_int = pix_int.copy()
    for j, arr_row in enumerate(arr_weight_y, 0):
        for i, v in enumerate(arr_row, 0):
            pb_sobel_y_float += v*pix_float_border[j:j+h, i:i+w]
            # pb_sobel_y_int += v*pix_int_border[j:j+h, i:i+w]

    pb_sobel_x_int = np.round(pb_sobel_x_float).astype(np.int)
    pb_sobel_y_int = np.round(pb_sobel_y_float).astype(np.int)

    div_x_min = np.min(pb_sobel_x_int)
    div_x_max = np.max(pb_sobel_x_int)
    div_y_min = np.min(pb_sobel_y_int)
    div_y_max = np.max(pb_sobel_y_int)

    print("div_x_min: {}".format(div_x_min))
    print("div_x_max: {}".format(div_x_max))
    print("div_y_min: {}".format(div_y_min))
    print("div_y_max: {}".format(div_y_max))

    y_size = div_y_max-div_y_min+1
    x_size = div_x_max-div_x_min+1

    arr_direction_template = np.zeros((y_size, x_size), dtype=np.int)
    arr_magnitude_template = np.zeros((y_size, x_size), dtype=np.int)

    for j, dy in enumerate(range(div_y_min, div_y_max+1), 0):
        print("dy: {}".format(dy))
        for i, dx in enumerate(range(div_x_min, div_x_max+1), 0):
            direction = 0
            magnitude = 0

            abs_dy = abs(dy)
            abs_dx = abs(dx)

            if dy==0 and dx==0:
                continue
            elif dy==0:
                if dx>0:
                    angle = 0
                else:
                    angle = 180
            elif dx==0:
                if dy>0:
                    angle = 90
                else:
                    angle = 270
            elif dy>0 and dx>=0:
                if abs_dy < abs_dx:
                    angle = 0+np.arctan(abs_dy/abs_dx)*180/3.141592654
                else:
                    angle = 90-np.arctan(abs_dx/abs_dy)*180/3.141592654
            elif dy>=0 and dx<0:
                if abs_dy > abs_dx:
                    angle = 90+np.arctan(abs_dx/abs_dy)*180/3.141592654
                else:
                    angle = 180-np.arctan(abs_dy/abs_dx)*180/3.141592654
            elif dy<0 and dx<=0:
                if abs_dy < abs_dx:
                    angle = 180+np.arctan(abs_dy/abs_dx)*180/3.141592654
                else:
                    angle = 270-np.arctan(abs_dx/abs_dy)*180/3.141592654
            elif dy<=0 and dx>0:
                if abs_dy > abs_dx:
                    angle = 270+np.arctan(abs_dx/abs_dy)*180/3.141592654
                else:
                    angle = 360-np.arctan(abs_dy/abs_dx)*180/3.141592654

            if angle>=22.5*1 and angle<22.5*3:
                direction = 2
            elif angle>=22.5*3 and angle<22.5*5:
                direction = 3
            elif angle>=22.5*5 and angle<22.5*7:
                direction = 4
            elif angle>=22.5*7 and angle<22.5*9:
                direction = 5
            elif angle>=22.5*9 and angle<22.5*11:
                direction = 6
            elif angle>=22.5*11 and angle<22.5*13:
                direction = 7
            elif angle>=22.5*13 and angle<22.5*15:
                direction = 8
            else:
                direction = 1

            magnitude = int(np.sqrt(dy**2 + dx**2))

            arr_direction_template[j, i] = direction
            arr_magnitude_template[j, i] = magnitude

    pix_dir = arr_direction_colors[arr_direction_template]
    img_dir = Image.fromarray(pix_dir)
    img_dir.save(dir_path+'img_template_direction_border_amount_{}.png'.format(border_amount))


    pix_mag = (arr_magnitude_template.astype(np.float) / np.max(arr_magnitude_template) * 255.999).astype(np.int).astype(np.uint8)
    img_mag = Image.fromarray(pix_mag)
    img_mag.save(dir_path+'img_tempalte_magnitude_border_amount_{}.png'.format(border_amount))

    print("border_amount: {}".format(border_amount))
    print("- div_x_min: {}".format(div_x_min))
    print("- div_x_max: {}".format(div_x_max))
    print("- div_y_min: {}".format(div_y_min))
    print("- div_y_max: {}".format(div_y_max))


    def calc_direction_magnitude(dy, dx):
        return arr_direction_template[dy-div_y_min-1, dx-div_x_min-1], arr_magnitude_template[dy-div_y_min-1, dx-div_x_min-1]


    nfunc = np.frompyfunc(calc_direction_magnitude, 2, 2)
    arr_direction, arr_magnitude = nfunc(pb_sobel_y_int, pb_sobel_x_int)

    return pb_sobel_x_int, pb_sobel_y_int, arr_direction.astype(np.int), arr_magnitude.astype(np.int)


def gauss_blur(pix, border_size):
    pix_int = pix.astype(int)
    pix_int_border_h = np.hstack((pix_int[:, :1], )*border_size+(pix_int, )+(pix_int[:, -1:], )*border_size)
    pix_int_border = np.vstack((pix_int_border_h[:1, :], )*border_size+(pix_int_border_h, )+(pix_int_border_h[-1:, :], )*border_size)

    kernel = gkern(kernlen=border_size*2+1, nsig=2)
    kernel_int = (kernel / np.min(kernel)).astype(np.int)

    h, w = pix_int.shape

    pix_int_sum = np.zeros(pix_int.shape, dtype=np.int)
    for j, arr_row in enumerate(kernel_int, 0):
        for i, v in enumerate(arr_row, 0):
            pix_int_sum += v*pix_int_border[j:j+h, i:i+w]
    pix_gauss = (pix_int_sum.astype(np.float)/np.sum(kernel_int)*(255.999/255.)).astype(np.int).astype(np.uint8)
    return pix_gauss


if __name__ == '__main__':
    DIR_IMAGES = PATH_ROOT_DIR+'images/'
    if not os.path.exists(DIR_IMAGES):
        os.makedirs(DIR_IMAGES)

    FILE_PATH_ORIG = DIR_IMAGES+FILE_NAME_JPG

    assert os.path.exists(FILE_PATH_ORIG)

    DIR_PATH_ORIG = FILE_PATH_ORIG[:FILE_PATH_ORIG.rfind('/')+1]


    img = Image.open(FILE_PATH_ORIG)

    resize = 3
    w, h = img.size
    img = img.resize((w*resize, h*resize), resample=Image.LANCZOS)

    FILE_NAME_PNG = re.sub(r'\.png$', '', FILE_NAME_JPG)+'_resize_{}.png'.format(resize)
    # FILE_NAME_PNG = re.sub(r'\.jpg$', '', FILE_NAME_JPG)+'_resize_{}.png'.format(resize)
    FILE_PATH_PNG = DIR_IMAGES+FILE_NAME_PNG
    if not os.path.exists(FILE_PATH_PNG):
        img.save(FILE_PATH_PNG)

    pix_orig = np.array(img)
    pix = pix_orig.copy()
    h, w, d = pix.shape

    u, c = np.unique(pix_orig.reshape((-1, )).view('u1,u1,u1'), return_counts=True)
    u_mat = u.view('u1').reshape((-1, d))
    u_sum = np.sum(u_mat.astype('i8')**2, axis=1)

    uc_comb = np.empty((u.shape[0], ), dtype='i8,i8,u1,u1,u1')
    uc_comb['f0'] = c
    uc_comb['f1'] = u_sum
    uc_comb['f2'] = u_mat[:, 0]
    uc_comb['f3'] = u_mat[:, 1]
    uc_comb['f4'] = u_mat[:, 2]

    uc_comb_sort = np.sort(uc_comb)

    sys.exit()

    img_gray = img.convert('L')
    img_gray.save(DIR_IMAGES+FILE_NAME_PNG.replace('.png', '_grayscale.png'))
    
    pix_gray = np.array(img_gray)
    print("pix.shape: {}".format(pix.shape))


    def create_hist_256(u, c):
        assert u.dtype == np.dtype('uint8')
        c_new = np.zeros((256, ), dtype=c.dtype)
        c_new[u] = c
        return c_new

    l_c = [create_hist_256(*np.unique(pix[..., i], return_counts=True)) for i in range(0, 3)]

    xs = np.arange(0, 256)
    ys_r = l_c[0]
    ys_g = l_c[1]
    ys_b = l_c[2]

    # fig, axs = plt.subplots(nrows=3, ncols=1)
    # fig.set_title('Histogram of image')
    # axs[0].bar(xs, ys_r, color='#FF0000', width=1.)
    # axs[1].bar(xs, ys_g, color='#00FF00', width=1.)
    # axs[2].bar(xs, ys_b, color='#0000FF', width=1.)
    # plt.show(block=False)


    pix_all = np.zeros((8, h, w, d), dtype=np.uint8)
    pix_all[7, ...] = pix

    # create simple rgb reduces images!
    for bit in range(1, 8):
        amount = 2**bit
        
        arr_idx = np.arange(0, amount + 1) * (256 / amount)
        arr_idx_round = np.round(arr_idx).astype(np.int)

        arr_val = np.arange(0, amount) * (255 / (amount - 1))
        arr_val_round = np.round(arr_val).astype(np.int)
        
        print("bit: {}\n- arr_idx_round: {}\n- arr_val_round: {}".format(bit, arr_idx_round, arr_val_round))

        arr_map = np.zeros((256, ), dtype=np.uint8)
        for i1, i2, v in zip(arr_idx_round[:-1], arr_idx_round[1:], arr_val):
            arr_map[i1:i2] = v
        print("arr_map: {}".format(arr_map))

        pix2 = np.empty(pix.shape, dtype=np.uint8)
        pix2[..., 0] = arr_map[pix[..., 0]]
        pix2[..., 1] = arr_map[pix[..., 1]]
        pix2[..., 2] = arr_map[pix[..., 2]]

        pix_all[bit-1, ...] = pix2

        img2 = Image.fromarray(pix2)
        img2.save(DIR_IMAGES+FILE_NAME_PNG.replace('.png', '_c{}b{}.png'.format(pix2.shape[2], bit)))


    bits = 2
    pix = pix_all[bits-1]
    img = Image.fromarray(pix)
    img_gray = img.convert('L')
    img_gray.save(DIR_IMAGES+FILE_NAME_PNG.replace('.png', '_grayscale_c1b{}.png'.format(bits)))

    pix_gray = np.array(img_gray)


    border_amount = 1
    
    pb_sobel_x_int, pb_sobel_y_int, arr_direction, arr_magnitude = create_sobel_image_derivative(pix_gray, border_amount, dir_path=DIR_PATH_ORIG)
    
    max_abs_v_x = np.max(np.abs(pb_sobel_x_int))
    pix_sobel_x = (pb_sobel_x_int.astype(np.float)*127.999/max_abs_v_x+128).astype(np.int).astype(np.uint8)

    max_abs_v_y = np.max(np.abs(pb_sobel_y_int))
    pix_sobel_y = (pb_sobel_y_int.astype(np.float)*127.999/max_abs_v_y+128).astype(np.int).astype(np.uint8)

    pb_sobel_int = np.sqrt(pb_sobel_x_int**2+pb_sobel_y_int**2)
    max_abs_v_sobel = np.max(np.abs(pb_sobel_int))
    pix_sobel = (pb_sobel_y_int.astype(np.float)*255.999/max_abs_v_sobel).astype(np.int).astype(np.uint8)
    
    pix_dir = arr_direction_colors[arr_direction]
    img_dir = Image.fromarray(pix_dir)
    img_dir.save(DIR_PATH_ORIG+'img_direction_border_amount_{:02}.png'.format(border_amount))

    # max_abs_v_mag = np.max(np.abs(arr_magnitude))
    # pix_sobel_mag = (pb_sobel_y_int.astype(np.float)*255.999/max_abs_v_mag).astype(np.int).astype(np.uint8)

    img_sobel_x = Image.fromarray(pix_sobel_x)
    img_sobel_y = Image.fromarray(pix_sobel_y)
    img_sobel = Image.fromarray(pix_sobel)
    # img_sobel_mag = Image.fromarray(pix_sobel_mag)

    # print('save img_sobel_x')
    # img_sobel_x.save(DIR_PATH_ORIG+'img_sobel_x.png')
    # print('save img_sobel_y')
    # img_sobel_y.save(DIR_PATH_ORIG+'img_sobel_y.png')
    print('save img_sobel')
    img_sobel.save(DIR_PATH_ORIG+'img_sobel.png')


    # border_size_gauss = 2
    d_pix_gauss_blur = {}
    d_pix_gauss_blur_sobel = {}
    for border_size_gauss in range(5, 16):
    # for border_size_gauss in range(1, 5):
    # for border_size_gauss in range(1, 4):
        print("border_size_gauss: {}".format(border_size_gauss))

        pix_gauss_blur = gauss_blur(pix_gray, border_size_gauss)


        img_gauss_blur = Image.fromarray(pix_gauss_blur)

        img_gauss_blur.save(DIR_IMAGES+FILE_NAME_PNG.replace('.png', '_gauss_blur_border_size_{:02}.png'.format(border_size_gauss)))

        pb_sobel_x_int, pb_sobel_y_int, arr_direction, arr_magnitude = create_sobel_image_derivative(pix_gauss_blur, border_amount, dir_path=DIR_PATH_ORIG)
        
        max_abs_v_x = np.max(np.abs(pb_sobel_x_int))
        pix_sobel_x = (pb_sobel_x_int.astype(np.float)*127.999/max_abs_v_x+128).astype(np.int).astype(np.uint8)

        max_abs_v_y = np.max(np.abs(pb_sobel_y_int))
        pix_sobel_y = (pb_sobel_y_int.astype(np.float)*127.999/max_abs_v_y+128).astype(np.int).astype(np.uint8)

        pb_sobel_int = np.sqrt(pb_sobel_x_int**2+pb_sobel_y_int**2)
        max_abs_v_sobel = np.max(np.abs(pb_sobel_int))
        pix_sobel = (pb_sobel_y_int.astype(np.float)*255.999/max_abs_v_sobel).astype(np.int).astype(np.uint8)
        
        d_pix_gauss_blur[border_size_gauss] = pix_gauss_blur
        d_pix_gauss_blur_sobel[border_size_gauss] = pix_sobel

        pix_dir = arr_direction_colors[arr_direction]
        img_dir = Image.fromarray(pix_dir)
        img_dir.save(DIR_PATH_ORIG+'img_direction_border_amount_{:02}_border_size_gauss_{:02}.png'.format(border_amount, border_size_gauss))

        # max_abs_v_mag = np.max(np.abs(arr_magnitude))
        # pix_sobel_mag = (pb_sobel_y_int.astype(np.float)*255.999/max_abs_v_mag).astype(np.int).astype(np.uint8)

        img_sobel_x = Image.fromarray(pix_sobel_x)
        img_sobel_y = Image.fromarray(pix_sobel_y)
        img_sobel = Image.fromarray(pix_sobel)
        # img_sobel_mag = Image.fromarray(pix_sobel_mag)

        # print('save img_gauss_blur_sobel_x')
        # img_sobel_x.save(DIR_PATH_ORIG+'img_gauss_blur_border_size_{:02}_sobel_x.png'.format(border_size_gauss))
        # print('save img_gauss_blur_sobel_y')
        # img_sobel_y.save(DIR_PATH_ORIG+'img_gauss_blur_border_size_{:02}_sobel_y.png'.format(border_size_gauss))
        print('save img_gauss_blur_sobel')
        img_sobel.save(DIR_PATH_ORIG+'img_gauss_blur_border_size_{:02}_sobel.png'.format(border_size_gauss))

        for threashold in [64, 128, 196]:
            idxs_contur = d_pix_gauss_blur_sobel[border_size_gauss]>=threashold
            pix_ = pix.copy()
            pix_[idxs_contur] = (0x00, 0x00, 0x00)
            img = Image.fromarray(pix_)
            img.save(DIR_PATH_ORIG+'img_add_black_outlines_gauss_blur_size_{}_bits_{}_threshold_{:03}.png'.format(border_size_gauss, bits, threashold))
            # img.show()
