import numpy as np

def bin_to_bpp4_idx(pix_bin):
    pix_bin = pix_bin.reshape((-1, 0x20))
    pix_idx = np.zeros((0, pix_bin.shape[0]), dtype=np.int)
    for i in range(0, 8):
        for j in range(7, -1, -1):
            pix_idx = np.vstack((pix_idx, (((pix_bin[:, i*2+0x00]>>j)&0x1)<<0)|
                                                (((pix_bin[:, i*2+0x01]>>j)&0x1)<<1)|
                                                (((pix_bin[:, i*2+0x10]>>j)&0x1)<<2)|
                                                (((pix_bin[:, i*2+0x11]>>j)&0x1)<<3)))
    return pix_idx.T.reshape((-1, 8, 8)).astype(np.uint8)


def bpp4_idx_to_bin(pix_idx):
    pix_idx = pix_idx.reshape((-1, 0x40))
    pix_bin = np.zeros((pix_idx.shape[0], 0x20), dtype=np.uint8)
    for i in range(0, 8):
        for j in range(0, 8):
            idx = pix_idx[:, i*8+j]
            shift = 7-j
            pix_bin[:, i*2+0x00] |= ((idx>>0)&0x1)<<(shift)
            pix_bin[:, i*2+0x01] |= ((idx>>1)&0x1)<<(shift)
            pix_bin[:, i*2+0x10] |= ((idx>>2)&0x1)<<(shift)
            pix_bin[:, i*2+0x11] |= ((idx>>3)&0x1)<<(shift)
    return pix_bin
