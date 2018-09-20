#! /usr/bin/python3.6

import os

import numpy as np

from PIL import Image

def create_some_random_pictures():
    # pix = np.zeros((256, 256, 3), dtype=np.uint8)
    pix = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    Image.fromarray(pix).save("images/test_random.png")
    pix_sort_x = np.sort(pix.view("u1,u1,u1"), axis=1).view("u1")
    pix_sort_y = np.sort(pix.view("u1,u1,u1"), axis=0).view("u1")

    Image.fromarray(pix_sort_x).save("images/test_random_sort_x.png")
    Image.fromarray(pix_sort_y).save("images/test_random_sort_y.png")

    pix_sort_x_row_y = np.sort(pix_sort_x.reshape((-1, )).view("u1,u1,u1"+",u1,u1,u1"*255)).view("u1").reshape((256, 256, 3))
    Image.fromarray(pix_sort_x_row_y).save("images/test_random_sort_x_row_y.png")

    # black and white
    pix_bw = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    pix_bw *= 255
    img_bw = Image.fromarray(pix_bw)
    img_bw.save("images/bw_random.png")

    pix_bw_sort_x = np.sort(pix_bw, axis=1)
    pix_bw_sort_y = np.sort(pix_bw, axis=0)
    Image.fromarray(pix_bw_sort_x).save("images/bw_random_sort_x.png")
    Image.fromarray(pix_bw_sort_y).save("images/bw_random_sort_y.png")

    if not os.path.exists("images/rng_sort"):
        os.makedirs("images/rng_sort")
    # convert r g b channels to bw from pix_sort_{x,y}
    for axis in ["x", "y"]:
        pix_sort = eval("pix_sort_{}".format(axis))
        for channel in range(0, 3):
            pix_sort_ax_ch = pix_sort[:, :, channel].copy()
            Image.fromarray(pix_sort_ax_ch).save("images/rng_sort/rnd_sort_{}_ch_{}_gray.png".format(axis, channel))
            pos = pix_sort_ax_ch < 128
            pix_sort_ax_ch[pos] = 0
            pix_sort_ax_ch[~pos] = 255
            Image.fromarray(pix_sort_ax_ch).save("images/rng_sort/rnd_sort_{}_ch_{}_bw.png".format(axis, channel))

if __name__ == "__main__":
    # create_some_random_pictures()
    
    height = 256
    width = height
    # width = 256
    path_pictures = "images/changing_bw_{}_{}/".format(height, width)
    if not os.path.exists(path_pictures):
        os.makedirs(path_pictures)

    pix_blank = np.zeros((height, width), dtype=np.uint8)
    pix_bw = np.random.randint(0, 2, (height, width), dtype=np.uint8)
    pix_bw *= 255
    Image.fromarray(pix_bw).save(path_pictures+"rnd_{}_{}_bw.png".format(height, width))

    # so long there are white pixels, repeat the elimination_process!
    it = 0
    pix_bw_prev = pix_bw.copy()
    pixs = [pix_bw.copy()]
    while np.sum(pix_bw == 255) > 0:
        print("it: {}".format(it))
        
        pos = np.where(pix_bw == 255)
        print("len(pos[0]): {}".format(len(pos[0])))

        pix_1 = pix_blank.copy()
        pix_2 = pix_blank.copy()
        pix_3 = pix_blank.copy()
        pix_4 = pix_blank.copy()

        pix_1[(lambda x: (pos[0][x]-1, pos[1][x]))(pos[0]>0)] += 1
        pix_2[(lambda x: (pos[0][x]+1, pos[1][x]))(pos[0]<(height-1))] += 1
        pix_3[(lambda x: (pos[0][x], pos[1][x]-1))(pos[1]>0)] += 1
        pix_4[(lambda x: (pos[0][x], pos[1][x]+1))(pos[1]<(width-1))] += 1

        # pix_bw = ((pix_1+pix_2+pix_3+pix_4+pix_bw) > 2)*255
        pix_bw = (pix_1&pix_2|pix_3&pix_4)*255
        # pix_bw = (((pix_1+pix_2+pix_3+pix_4)>2)+0)*255
        # pix_bw = (((pix_1+pix_2+pix_3+pix_4+(pix_bw==255))>3)+0)*255
        # pix_bw = ((pix_1+(pix_bw==255)) > 2)*255

        pix_bw = pix_bw.astype(np.uint8)

        # for y, x in zip(*pos):
        #     s = 0
        #     if ((y > 0) and (pix_bw[y-1, x] == 0)):
        #         s += 1
        #     if ((y < height-1) and (pix_bw[y+1, x] == 0)):
        #         s += 1
        #     if ((x > 0) and (pix_bw[y, x-1] == 0)):
        #         s += 1
        #     if ((x < width-1) and (pix_bw[y, x+1] == 0)):
        #         s += 1
        #     if s >= 3:
        #         pix_bw[y, x] = 0

        if np.sum(pix_bw_prev != pix_bw) == 0:
            break

        is_found = False
        for pix in pixs:
            if np.sum(pix != pix_bw) == 0:
                is_found = True
                break
        if is_found:
            break

        pixs.append(pix_bw.copy())

        Image.fromarray(pix_bw).save(path_pictures+"rnd_{}_{}_bw_iter_{:03}.png".format(height, width, it))
        it += 1
        
        pix_bw_prev = pix_bw.copy()
