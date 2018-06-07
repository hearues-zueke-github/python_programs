#! /usr/bin/python3.5

import sys

import numpy as np

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
    mask_d = np.array([-1, -1, 0, -1, 0, 1, 0, 1, 1]).astype(np.int)
    mask_cd = np.array([0, -1, -1, 1, 0, -1, 1, 1, 0]).astype(np.int)

    pix_deriv_x = np.dot(pix_deriv_table, mask_x).reshape(shape)
    pix_deriv_y = np.dot(pix_deriv_table, mask_y).reshape(shape)
    pix_deriv_d = np.dot(pix_deriv_table, mask_d).reshape(shape)
    pix_deriv_cd = np.dot(pix_deriv_table, mask_cd).reshape(shape)

    return pix_deriv_x, pix_deriv_y, pix_deriv_d, pix_deriv_cd

def get_integral_image(pix, border_size):
    pix_border = padding_border(pix, border_size).astype(np.float)

    pix_border_1 = np.vstack((np.zeros((1, pix_border.shape[1])), pix_border))
    pix_border_2 = np.hstack((np.zeros((pix_border_1.shape[0], 1)), pix_border_1))

    pix_integral = np.cumsum(np.cumsum(pix_border_2, axis=1), axis=0)

    return pix_integral


if __name__ == "__main__":
    # img = Image.open("nature_1.jpg").convert('LA')
    img = Image.open("nature_1.jpg")

    # img.show()
    pix_orig = np.array(img)
    pix_int_gray = np.dot(pix_orig.astype(np.float)[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    print("pix_int_gray.shape: {}".format(pix_int_gray.shape))

    # Image.fromarray(pix_int_gray).show()
    # sys.exit(0)

    pix = pix_int_gray.astype(np.int)

    print("Calculate derivations in x and y with sobel!")
    pix_deriv_x, pix_deriv_y, pix_deriv_d, pix_deriv_cd = get_derivations_sobel(pix)

    pix_deriv_x = pix_deriv_x.astype(np.float)
    pix_deriv_y = pix_deriv_y.astype(np.float)
    pix_deriv_d = pix_deriv_d.astype(np.float)
    pix_deriv_cd = pix_deriv_cd.astype(np.float)

    deriv_max_x = np.max(np.abs(pix_deriv_x))
    deriv_max_y = np.max(np.abs(pix_deriv_y))
    deriv_max_d = np.max(np.abs(pix_deriv_d))
    deriv_max_cd = np.max(np.abs(pix_deriv_cd))

    pix_deriv_x /= deriv_max_x
    pix_deriv_y /= deriv_max_y
    pix_deriv_d /= deriv_max_d
    pix_deriv_cd /= deriv_max_cd

    pix_deriv_x = pix_deriv_x*128+128
    pix_deriv_y = pix_deriv_y*128+128
    pix_deriv_d = pix_deriv_d*128+128
    pix_deriv_cd = pix_deriv_cd*128+128

    pix_deriv_x_int = pix_deriv_x.astype(np.uint8)
    pix_deriv_y_int = pix_deriv_y.astype(np.uint8)
    pix_deriv_d_int = pix_deriv_d.astype(np.uint8)
    pix_deriv_cd_int = pix_deriv_cd.astype(np.uint8)

    pix_deriv_x_int[pix_deriv_x > 255.5] = 255
    pix_deriv_y_int[pix_deriv_y > 255.5] = 255
    pix_deriv_d_int[pix_deriv_d > 255.5] = 255
    pix_deriv_cd_int[pix_deriv_cd > 255.5] = 255

    img = Image.fromarray(pix_int_gray)
    img_deriv_x = Image.fromarray(pix_deriv_x_int)
    img_deriv_y = Image.fromarray(pix_deriv_y_int)
    img_deriv_d = Image.fromarray(pix_deriv_d_int)
    img_deriv_cd = Image.fromarray(pix_deriv_cd_int)

    print("Save all images!")
    img.save("img.png", "PNG")
    img_deriv_d.save("img_deriv_1_d.png", "PNG")
    img_deriv_y.save("img_deriv_2_y.png", "PNG")
    img_deriv_cd.save("img_deriv_3_cd.png", "PNG")
    img_deriv_x.save("img_deriv_4_x.png", "PNG")

    pix_integral = get_integral_image(pix_int_gray, 200)

    pix_integral_norm = pix_integral/np.max(pix_integral)*256
    pix_integral_int = pix_integral_norm.astype(np.uint8)
    pix_integral_int[pix_integral_norm > 255.5] = 255

    img_integral = Image.fromarray(pix_integral_int)
    img_integral.save("img_integral.png", "PNG")

    # TODO: make the box derivations more generic!