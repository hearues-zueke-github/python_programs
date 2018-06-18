# bs...border_size
def get_derivatives_box(pix, pix_integral, bs, s=1):
    # First do the x-derivative
    # h...height
    # w...width
    h, w = pix.shape
    
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

    print("Needed time for deriv other: {:.4f}".format(end_time-start_time))

    return deriv_x, deriv_y

def get_derivatives_box_tiles(tile_h, tile_w, pix_integral, bs, s=1, offset=0):
    # First do the x-derivative
    # h...height
    # w...width
    h, w = tile_h, tile_w
    # print("bs: {}".format(bs))
    # print("pix.shape: {}".format(pix.shape))
    # offset = 1
    bs_l = bs-offset
    bs_h = bs+offset
    
    v_1 = pix_integral[:, bs_l-s  :bs_h+h-s  , bs_l+1  :bs_h+w+1]
    v_2 = pix_integral[:, bs_l+s+1:bs_h+h+s+1, bs_l+1+s:bs_h+w+1+s]
    v_3 = pix_integral[:, bs_l-s  :bs_h+h-s  , bs_l+1+s:bs_h+w+1+s]
    v_4 = pix_integral[:, bs_l+s+1:bs_h+h+s+1, bs_l+1  :bs_h+w+1]
    print("v_1.shape: {}".format(v_1.shape))
    print("v_2.shape: {}".format(v_2.shape))
    print("v_3.shape: {}".format(v_3.shape))
    print("v_4.shape: {}".format(v_4.shape))
    deriv_x_right = v_1+v_2-v_3-v_4
    v_1 = pix_integral[:, bs_l-s  :bs_h+h-s  , bs_l-s  :bs_h+w-s]
    v_2 = pix_integral[:, bs_l+s+1:bs_h+h+s+1, bs_l    :bs_h+w]
    v_3 = pix_integral[:, bs_l-s  :bs_h+h-s  , bs_l    :bs_h+w]
    v_4 = pix_integral[:, bs_l+s+1:bs_h+h+s+1, bs_l-s  :bs_h+w-s]
    deriv_x_left  = v_1+v_2-v_3-v_4

    deriv_y_right = pix_integral[:, bs_l+1  :bs_h+h+1  , bs_l-s  :bs_h+w-s  ] \
                   +pix_integral[:, bs_l+1+s:bs_h+h+1+s, bs_l+s+1:bs_h+w+s+1] \
                   -pix_integral[:, bs_l+1+s:bs_h+h+1+s, bs_l-s  :bs_h+w-s  ] \
                   -pix_integral[:, bs_l+1  :bs_h+h+1  , bs_l+s+1:bs_h+w+s+1]
    deriv_y_left  = pix_integral[:, bs_l-s  :bs_h+h-s  , bs_l-s  :bs_h+w-s  ] \
                   +pix_integral[:, bs_l    :bs_h+h    , bs_l+s+1:bs_h+w+s+1] \
                   -pix_integral[:, bs_l    :bs_h+h    , bs_l-s  :bs_h+w-s  ] \
                   -pix_integral[:, bs_l-s  :bs_h+h-s  , bs_l+s+1:bs_h+w+s+1]

    deriv_x = deriv_x_right-deriv_x_left
    deriv_y = deriv_y_right-deriv_y_left

    return deriv_x, deriv_y
