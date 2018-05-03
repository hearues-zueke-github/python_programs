#! /usr/bin/python2.7

import numpy as np

from PIL import Image

if __name__ == "__main__":
    height = 300
    width = 300

    get_new_binary_pix = lambda: np.random.randint(0, 2, (height, width, 3)).astype(np.uint8)
    
    pix = get_new_binary_pix()

    img = Image.fromarray(pix*255)
    img.show()

    pix_factors = get_new_binary_pix()

    pix[:, :, 0] = np.dot(pix[:, :, 0], pix_factors[:, :, 0])
    pix[:, :, 1] = np.dot(pix[:, :, 1], pix_factors[:, :, 1])
    pix[:, :, 2] = np.dot(pix[:, :, 2], pix_factors[:, :, 2])
    
    pix %= 2
    
    img_2 = Image.fromarray(pix*255)
    img_2.show()
