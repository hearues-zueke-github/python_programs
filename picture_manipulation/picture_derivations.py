#! /usr/bin/python3.5

import os
import sys

import numpy as np

from time import time

from PIL import Image

def padding_border(pix, border_size=1):
    def get_padded_rows(row_1, row_2):
        row_pad_1 = np.zeros((border_size, row_1.shape[0])).astype(pix.dtype)
        row_pad_2 = np.zeros((border_size, row_1.shape[0])).astype(pix.dtype)

        row_pad_1[:] = row_1
        row_pad_2[:] = row_2

        return row_pad_1, row_pad_2

    row_pad_1, row_pad_2 = get_padded_rows(pix[0], pix[-1])
    pix_1 = np.vstack((row_pad_1, pix, row_pad_2)).T

    row_pad_1, row_pad_2 = get_padded_rows(pix_1[0], pix_1[-1])
    pix_2 = np.vstack((row_pad_1, pix_1, row_pad_2)).T

    return pix_2

def get_derivations_sobel(pix):
    shape = pix.shape

    # kernel size is 3 as default, but 5, 7, etc would also work too!
    start_time = time()
    kernel_size = 3
    pix_pad = padding_border(pix, kernel_size//2)

    idx_y, idx_x = np.array([(j, i) for j in range(0, kernel_size) for i in range(0, kernel_size)]).T

    rows, cols = shape

    idx_y_tbl = np.zeros((rows, cols)).astype(np.int)
    idx_y_tbl[:] = np.arange(0, rows).reshape((-1, 1))
    idx_x_tbl = np.zeros((rows, cols)).astype(np.int)
    idx_x_tbl[:] = np.arange(0, cols).reshape((1, -1))

    idx_y_row = np.add.outer(idx_y_tbl, idx_y).reshape((-1, ))
    idx_x_row = np.add.outer(idx_x_tbl, idx_x).reshape((-1, ))

    print("Now here! Yeah!")

    pix_deriv_table = pix_pad[idx_y_row, idx_x_row].reshape((-1, kernel_size**2))

    mask_sobel = np.zeros((kernel_size, kernel_size)).astype(np.int)
    mask_sobel[0] = -1
    mask_sobel[-1] = 1
    mask_x = mask_sobel.T.copy().reshape((-1, ))
    mask_y = mask_sobel.copy().reshape((-1, ))

    pix_deriv_x = np.dot(pix_deriv_table, mask_x).reshape(shape)
    pix_deriv_y = np.dot(pix_deriv_table, mask_y).reshape(shape)

    end_time = time()
    print("Needed time for deriv normal: {:.4f}".format(end_time-start_time))
    
    mask_d = np.array([-1, -1, 0, -1, 0, 1, 0, 1, 1]).astype(np.int)
    mask_cd = np.array([0, -1, -1, 1, 0, -1, 1, 1, 0]).astype(np.int)
    
    pix_deriv_d = np.dot(pix_deriv_table, mask_d).reshape(shape)
    pix_deriv_cd = np.dot(pix_deriv_table, mask_cd).reshape(shape)

    return pix_deriv_x, pix_deriv_y, pix_deriv_d, pix_deriv_cd

def get_integral_image(pix, border_size):
    pix_border = padding_border(pix, border_size).astype(np.float)

    pix_border_1 = np.vstack((np.zeros((1, pix_border.shape[1])), pix_border))
    pix_border_2 = np.hstack((np.zeros((pix_border_1.shape[0], 1)), pix_border_1))

    pix_integral = np.cumsum(np.cumsum(pix_border_2, axis=1), axis=0)

    return pix_integral

# bs...border_size
def get_derivatives_box(pix, pix_integral, bs, s=1):
    # First do the x-derivative
    # h...height
    # w...width
    h, w = pix.shape
    
    start_time = time()
    deriv_x_right = pix_integral[bs-s  :bs+h-s  , bs+1  :bs+w+1] \
                   +pix_integral[bs+s+1:bs+h+s+1, bs+1+s:bs+w+1+s] \
                   -pix_integral[bs-s  :bs+h-s  , bs+1+s:bs+w+1+s] \
                   -pix_integral[bs+s+1:bs+h+s+1, bs+1  :bs+w+1]
    deriv_x_left  = pix_integral[bs-s  :bs+h-s  , bs-s  :bs+w-s] \
                   +pix_integral[bs+s+1:bs+h+s+1, bs    :bs+w] \
                   -pix_integral[bs-s  :bs+h-s  , bs    :bs+w] \
                   -pix_integral[bs+s+1:bs+h+s+1, bs-s  :bs+w-s]

    deriv_y_right = pix_integral[bs+1  :bs+h+1  , bs-s  :bs+w-s  ] \
                   +pix_integral[bs+1+s:bs+h+1+s, bs+s+1:bs+w+s+1] \
                   -pix_integral[bs+1+s:bs+h+1+s, bs-s  :bs+w-s  ] \
                   -pix_integral[bs+1  :bs+h+1  , bs+s+1:bs+w+s+1]
    deriv_y_left  = pix_integral[bs-s  :bs+h-s  , bs-s  :bs+w-s  ] \
                   +pix_integral[bs    :bs+h    , bs+s+1:bs+w+s+1] \
                   -pix_integral[bs    :bs+h    , bs-s  :bs+w-s  ] \
                   -pix_integral[bs-s  :bs+h-s  , bs+s+1:bs+w+s+1]

    deriv_x = deriv_x_right-deriv_x_left
    deriv_y = deriv_y_right-deriv_y_left

    end_time = time()

    print("Needed time for deriv other: {:.4f}".format(end_time-start_time))

    return deriv_x, deriv_y

if __name__ == "__main__":
    from os.path import expanduser
    home = expanduser("~")
    print("home: {}".format(home))

    # img = Image.open("nature_1.jpg").convert('LA')
    # img = Image.open(home+"/Pictures/picture_manipulation/nature_2.jpg")
    img = Image.open(home+"/Pictures/picture_manipulation/scanned_puzzels_nr_1.png")
    # img = Image.open("nature_1.jpg")

    # img.show()
    pix_orig = np.array(img)
    pix_int_gray = np.dot(pix_orig.astype(np.float)[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    print("pix_int_gray.shape: {}".format(pix_int_gray.shape))

    # Image.fromarray(pix_int_gray).show()
    # sys.exit(0)

    pix = pix_int_gray.astype(np.int)

    print("Calculate derivations in x and y with sobel!")
    pix_deriv_x, pix_deriv_y, pix_deriv_d, pix_deriv_cd = get_derivations_sobel(pix)

    def get_normalized_derivate(pix_deriv):
        pix_deriv = pix_deriv.copy().astype(np.float)
        deriv_max = np.max(np.abs(pix_deriv))
        pix_deriv /= deriv_max
        pix_deriv = pix_deriv*128+128
        pix_deriv_int = pix_deriv.astype(np.uint8)
        pix_deriv_int[pix_deriv > 255.5] = 255

        return pix_deriv_int

    def get_edge_image(pix_deriv, threshold):
        pix_deriv_abs = np.abs(pix_deriv)
        pix_edge = np.zeros(pix_deriv.shape).astype(np.uint8)
        pix_edge[pix_deriv_abs >= threshold] = 255

        return pix_edge

    pix_deriv_x_int = get_normalized_derivate(pix_deriv_x)
    pix_deriv_y_int = get_normalized_derivate(pix_deriv_y)
    pix_deriv_d_int = get_normalized_derivate(pix_deriv_d)
    pix_deriv_cd_int = get_normalized_derivate(pix_deriv_cd)

    # print("pix_deriv_x_int.shape: {}".format(pix_deriv_x_int.shape))

    img = Image.fromarray(pix_int_gray)
    img_deriv_x = Image.fromarray(pix_deriv_x_int)
    img_deriv_y = Image.fromarray(pix_deriv_y_int)
    img_deriv_d = Image.fromarray(pix_deriv_d_int)
    img_deriv_cd = Image.fromarray(pix_deriv_cd_int)
    
    folder_derivatives = home+"/Pictures/image_derivatives/"
    if not os.path.exists(folder_derivatives):
        os.makedirs(folder_derivatives)

    print("Save all images!")
    img.save(folder_derivatives+"img.png", "PNG")
    img_deriv_d.save(folder_derivatives+"img_deriv_1_d.png", "PNG")
    img_deriv_y.save(folder_derivatives+"img_deriv_2_y.png", "PNG")
    img_deriv_cd.save(folder_derivatives+"img_deriv_3_cd.png", "PNG")
    img_deriv_x.save(folder_derivatives+"img_deriv_4_x.png", "PNG")

    folder_edges = home+"/Pictures/image_edges/"
    if not os.path.exists(folder_edges):
        os.makedirs(folder_edges)

    # for threshold in np.arange(25, 101, 25):
    #     print("threshold: {}".format(threshold))
    #     img_deriv_x_edge = Image.fromarray(get_edge_image(pix_deriv_x, threshold))
    #     img_deriv_y_edge = Image.fromarray(get_edge_image(pix_deriv_y, threshold))
    #     img_deriv_d_edge = Image.fromarray(get_edge_image(pix_deriv_d, threshold))
    #     img_deriv_cd_edge = Image.fromarray(get_edge_image(pix_deriv_cd, threshold))

    #     img_deriv_d_edge.save(folder_edges+"img_deriv_edge_1_d_threshold_{:03}.png".format(threshold), "PNG")
    #     img_deriv_y_edge.save(folder_edges+"img_deriv_edge_2_y_threshold_{:03}.png".format(threshold), "PNG")
    #     img_deriv_cd_edge.save(folder_edges
    #         +"img_deriv_edge_3_cd_threshold_{:03}.png".format(threshold), "PNG")
    #     img_deriv_x_edge.save(folder_edges+"img_deriv_edge_4_x_threshold_{:03}.png".format(threshold), "PNG")

    border_size = 200
    pix_integral = get_integral_image(pix_int_gray, border_size)

    pix_integral_norm = pix_integral/np.max(pix_integral)*256
    pix_integral_int = pix_integral_norm.astype(np.uint8)
    pix_integral_int[pix_integral_norm > 255.5] = 255

    img_integral = Image.fromarray(pix_integral_int)
    img_integral.save("img_integral.png", "PNG")

    # Do the derivative withother function
    print("Do the derivates with other function!")
    pix_deriv_x_other, pix_deriv_y_other = get_derivatives_box(pix, pix_integral, border_size, s=2)
    
    pix_deriv_x_other_int = get_normalized_derivate(pix_deriv_x_other)
    img_deriv_x_other = Image.fromarray(pix_deriv_x_other_int)
    img_deriv_x_other.save(folder_derivatives+"img_deriv_other_x.png", "PNG")
    
    pix_deriv_y_other_int = get_normalized_derivate(pix_deriv_y_other)
    img_deriv_y_other = Image.fromarray(pix_deriv_y_other_int)
    img_deriv_y_other.save(folder_derivatives+"img_deriv_other_y.png", "PNG")

    # TODO: make the box derivations more generic!

    pix_deriv_x_xor = pix_deriv_x_int ^ pix_deriv_x_other_int
    pix_deriv_x_xor[pix_deriv_x_xor > 0] = 255
    img_deriv_x_xor = Image.fromarray(pix_deriv_x_xor)
    img_deriv_x_xor.save(folder_derivatives+"img_deriv_x_xor.png", "PNG")

    pix_deriv_y_xor = pix_deriv_y_int ^ pix_deriv_y_other_int
    pix_deriv_y_xor[pix_deriv_y_xor > 0] = 255
    img_deriv_y_xor = Image.fromarray(pix_deriv_y_xor)
    img_deriv_y_xor.save(folder_derivatives+"img_deriv_y_xor.png", "PNG")
