#! /usr/bin/python3.6

import os
import sys

import numpy as np

from PIL import Image

def get_all_rbgs():
    rgbs = np.zeros((16, 16, 256, 256, 3), dtype=np.uint8)

    values_2d = np.zeros((256, 256), dtype=np.uint8)
    values_2d[:] = np.arange(0, 256)

    rgbs[:, :, :, :, 2] = values_2d
    rgbs[:, :, :, :, 1] = values_2d.T

    for y in range(0, 16):
        for x in range(0, 16):
            rgbs[y, x, :, :, 0] = y*16+x

    return rgbs.reshape((-1, 3))

def get_new_rgbs_table(rgbs):
    rgbs_sum = (np.sum(rgbs.astype(np.int)*256**np.arange(2, -1, -1), axis=1)) >> 1
    return ((rgbs_sum.reshape((-1, 1))//256**np.arange(2, -1, -1)) % 256).astype(np.uint8)

def find_best_approx_array(arr_1, arr_2):
    assert isinstance(arr_1, np.ndarray)
    assert isinstance(arr_2, np.ndarray)

    assert len(arr_1.shape) == 2
    assert len(arr_2.shape) == 2
    assert arr_1.shape == arr_2.shape

    n, m = arr_1.shape

    euclid_dist = np.sum((arr_1.reshape((n, 1, m))-arr_2)**2, axis=-1)
    temp_1 = np.vstack((np.argmin(euclid_dist, axis=1), np.arange(0, n)))
    temp_2 = temp_1.T.reshape((-1, )).view("i8,i8")
    best_idx = np.sort(temp_2, order=["f0"]).view("i8").reshape((-1, 2)).T[1]

    return arr_1[best_idx]

if __name__ == "__main__":
    rgbs = get_all_rbgs()

    get_list_rgb_tuples = lambda rgbs: rgbs.view("u1,u1,u1").reshape((-1, )).tolist()

    # rgbs_new = get_new_rgbs_table(rgbs)

    path_images = "images/"
    # file_name = "pexels-photo-236047.jpeg"
    file_name = "fall-autumn-red-season.jpg"

    path_images_manipulated = "images/image_manipulated/"
    if not os.path.exists(path_images_manipulated):
        os.makedirs(path_images_manipulated)
    file_name_template = file_name.replace(".jpg", "_nr_{:02}.jpg")
    # file_name_template = file_name.replace(".jpeg", "_nr_{:02}.jpeg")
    
    img = Image.open(path_images+file_name)
    pix = np.array(img)
    shape = pix.shape

    # get every idx of each pixel
    pix_rgb_idxs = np.sum(pix.astype(np.int)*256**np.arange(2, -1, -1), axis=-1)
    
    # first get all unique pixels from image
    rgb_unique = np.unique(pix.view("u1,u1,u1").reshape((-1, ))).view("u1").reshape((-1, 3))
    # now get all idxs of each unique rgb
    rgb_unique_idxs = np.sum(rgb_unique.astype(np.int)*256**np.arange(2, -1, -1), axis=1)

    rgb_changes = rgb_unique.copy()

    Image.fromarray(pix).save(path_images_manipulated+file_name_template.format(0))
    for i in range(1, 100):
        print("Image: i: {}".format(i))
        # rgbs_idx = np.sum(pix.astype(np.int).reshape((-1, 3))*256**np.arange(2, -1, -1), axis=1)
        
        # Instead of permutation, choose 2 n amount of piles (e.g. 1st pile 1000 rgbs,
        # 2nd pile 1000 rgbs). Then approx for each rgb in 1st pile the best position
        # in 2nd pile, and vice verse.

        # rgb_unique stays always the same!
        amount = rgb_changes.shape[0]
        idx_mix = np.random.permutation(np.arange(0, amount))

        n = 10000
        # now take idx for 1st and 2nd pile
        idx_mix_1 = idx_mix[0:n]
        idx_mix_2 = idx_mix[n:2*n]

        # and get the 2 piles!
        rgb_pile_1 = rgb_changes[idx_mix_1].copy()
        rgb_pile_2 = rgb_changes[idx_mix_2].copy()

        # now always take as the reference color from the rgb_unique array!
        rgb_unique_pile_1 = rgb_unique[idx_mix_1]
        rgb_unique_pile_2 = rgb_unique[idx_mix_2]

        rgb_pile_1_approx = find_best_approx_array(rgb_pile_1, rgb_unique_pile_2)
        rgb_pile_2_approx = find_best_approx_array(rgb_pile_2, rgb_unique_pile_1)

        rgbs[rgb_unique_idxs[idx_mix_2]] = rgb_pile_1_approx
        rgbs[rgb_unique_idxs[idx_mix_1]] = rgb_pile_2_approx

        # rgbs[idxs] = rgb_changes
        pix_new = rgbs[pix_rgb_idxs].copy().astype(np.uint8)
        # print("np.sum(pix_new!=[0,0,0]: {}".format(np.sum(pix_new!=[0,0,0])))
        Image.fromarray(pix_new).save(path_images_manipulated+file_name_template.format(i))
        # pix = pix_new
