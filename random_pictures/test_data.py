#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import gzip

import numpy as np

if __name__ == "__main__":
    print("Loading data to temp obj!")
    with gzip.open("/tmp/temp_obj.pkl.gz", "rb") as f:
        d = dill.load(f)

    pixses_src_rgb = d['pixses_src_rgb']
    pixses_rgb_smaller = d['pixses_rgb_smaller']

    pixses_rgb_smaller = pixses_rgb_smaller.astype(np.int64)
    pixses_src_rgb = pixses_src_rgb.astype(np.int64)
    
    p = pixses_rgb_smaller[0].astype(np.int64)
    print("p.shape: {}".format(p.shape))

    idxs_y = np.zeros((45, 60), dtype=np.int64)+np.arange(0, 45).reshape((-1, 1)).astype(np.int64)
    idxs_x = np.zeros((45, 60), dtype=np.int64)+np.arange(0, 60).reshape((1, -1)).astype(np.int64)

    idxs = np.dstack((idxs_y, idxs_x))
    print("idxs.shape: {}".format(idxs.shape))

    idxs_parts = ( idxs
        .reshape((45//5, 5, 60, 2))
        .transpose(0, 2, 1, 3)
        .reshape((45//5*60//5, 5, 5, 2))
        .transpose(0, 2, 1, 3)
    )

    idxs_1d_y, idxs_1d_x = idxs_parts.reshape((-1, 2)).T

    print("idxs_1d_y: {}".format(idxs_1d_y))
    print("idxs_1d_x: {}".format(idxs_1d_x))

    def get_feature_matrix(pixses):    
        m_row_sum_feature = np.sum(pixses, axis=1).transpose(0, 2, 1).reshape((pixses.shape[0], -1))
        m_col_sum_feature = np.sum(pixses, axis=2).transpose(0, 2, 1).reshape((pixses.shape[0], -1))
        m_5x5_feature = np.sum(pixses[:, idxs_1d_y, idxs_1d_x].reshape((pixses.shape[0], -1, 5*5, 3)), axis=2).transpose(0, 2, 1).reshape((pixses.shape[0], -1))

        m_feature = np.hstack((m_row_sum_feature, m_col_sum_feature, m_5x5_feature))

        return m_feature

    m_rgb_feature = get_feature_matrix(pixses_rgb_smaller)
    m_src_feature = get_feature_matrix(pixses_src_rgb)

    v_row_sum_feature = np.sum(p, axis=0).T.reshape((-1, ))
    v_col_sum_feature = np.sum(p, axis=1).T.reshape((-1, ))
    v_5x5_feature = np.sum(p[idxs_1d_y, idxs_1d_x].reshape((-1, 5*5, 3)), axis=1).T.reshape((-1, ))

    v_feature = np.hstack((v_row_sum_feature, v_col_sum_feature, v_5x5_feature))


    # def test_5x5_feature(p):
    #     v_5x5_feature_own = np.zeros((12*9*3, ), dtype=np.int64)
    #     for c in range(0, 3):
    #         for y in range(0, 9):
    #             for x in range(0, 12):
    #                 v_5x5_feature_own[c*9*12+y*12+x] = np.sum(p[5*y:5*(y+1), 5*x:5*(x+1), c])
    #     return v_5x5_feature_own
    # v_5x5_feature_own = test_5x5_feature(p)
