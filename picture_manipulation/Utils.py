import numpy as np

"""
    @param pix: numpay array of a image
    @param tw: tile width/height
    @return: returns a numpay array with the squared tiles
"""
def get_square_tiles(pix, tw):
    x, y, z = pix.shape
    return pix.transpose(0, 2, 1) \
              .reshape((x//tw, tw*z, y)) \
              .transpose(0, 2, 1) \
              .reshape((x*y//tw//tw, tw, tw, z)) \
              .transpose(0, 2, 1, 3) \
              .reshape((x//tw, y//tw, tw, tw, z)) # this is crazy!

"""
    @param pix: numpay array of a image
    @param tw: tile width/height
    @return: returns a tuple
             1st value: a numpay array with the pixelated tiles
             2nd value: the rgb pixel map for the picture
"""
def get_pixelated_pix(pix, tw):
    height, width, _ = pix.shape
    pix_crop = pix[:(height//tw)*tw, :(width//tw)*tw]
    pix_crop_tiles = get_square_tiles(pix_crop, tw)
    pix_crop_tiles_rgb = np.mean(np.mean(pix_crop_tiles, axis=-2), axis=-2).astype(np.uint8)

    pix_crop_pixel = np.zeros(pix_crop.shape, dtype=pix_crop.dtype)
    for y in range(0, pix_crop_tiles.shape[0]):
        for x in range(0, pix_crop_tiles.shape[1]):
            pix_crop_pixel[tw*y:tw*(y+1), tw*x:tw*(x+1)] = pix_crop_tiles_rgb[y, x]

    return pix_crop_pixel, pix_crop_tiles_rgb

def get_pixelated_tiled_pix(pix, tw, pix_tiles_line):
    height, width, _ = pix.shape
    height_crop = (height//tw)*tw
    width_crop = (width//tw)*tw
    pix_crop = pix[:height_crop, :width_crop]
    pix_crop_tiles = get_square_tiles(pix_crop, tw)

    get_mean_mean = lambda pix: np.mean(np.mean(pix, axis=-2), axis=-2)
    get_sum_rgb = lambda pix_rgb: np.sum(pix_rgb*256**np.arange(0, 3), axis=-1)
    get_norm_pix = lambda pix: (lambda pix, min_val, max_val: (pix-min_val)*256**3/(max_val-min_val))(
        *(pix, np.min(pix), np.max(pix)))
    
    pix_crop_tiles_rgb = get_mean_mean(pix_crop_tiles)
    pix_tiles_line_rgb = get_mean_mean(pix_tiles_line)
    
    pix_crop_tiles_sum = get_sum_rgb(pix_crop_tiles_rgb)
    pix_tiles_line_sum = get_sum_rgb(pix_tiles_line_rgb)

    pix_crop_tiles_norm = get_norm_pix(pix_crop_tiles_sum)
    pix_tiles_line_norm = get_norm_pix(pix_tiles_line_sum)

    norm_values = np.arange(0, pix_tiles_line.shape[0]+1)/(pix_tiles_line.shape[0]+1)*256**3

    pix_crop_tiles_norm = pix_crop_tiles_norm.reshape(pix_crop_tiles_norm.shape+(1, ))

    idx_table = np.argmin(np.abs(pix_crop_tiles_norm-pix_tiles_line_norm), axis=-1)

    pix_crop_filled = np.zeros((height_crop, width_crop, pix.shape[2]), dtype=pix.dtype)
    for y in range(pix_crop_tiles_norm.shape[0]):
        for x in range(pix_crop_tiles_norm.shape[1]):
            pix_crop_filled[tw*y:tw*(y+1), tw*x:tw*(x+1)] = pix_tiles_line[idx_table[y, x]]

    # print("pix_crop_tiles_rgb.shape: {}".format(pix_crop_tiles_rgb.shape))
    # print("pix_tiles_line_rgb.shape: {}".format(pix_tiles_line_rgb.shape))
    # print("pix_crop_tiles_sum.shape: {}".format(pix_crop_tiles_sum.shape))
    # print("pix_tiles_line_sum.shape: {}".format(pix_tiles_line_sum.shape))
    # print("pix_crop_tiles_norm.shape: {}".format(pix_crop_tiles_norm.shape))
    # print("pix_tiles_line_norm.shape: {}".format(pix_tiles_line_norm.shape))

    return pix_crop_filled
    # return (pix_crop_tiles_rgb,
    #         pix_tiles_line_rgb,
    #         pix_crop_tiles_sum,
    #         pix_tiles_line_sum,
    #         pix_crop_tiles_norm,
    #         pix_tiles_line_norm)# pix_crop_tiles_norm, pix_tiles_line_norm
