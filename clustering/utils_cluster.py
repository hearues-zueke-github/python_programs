from typing import List, Dict, Set, Mapping, Any, Tuple

import matplotlib.pyplot as plt

from dotmap import DotMap
import numpy as np

__version__ = '0.1.0'
dm_obj_file_name = 'dm_obj.pkl.gz'

# l_color = [
#     '#00F020',
#     '#008000',
#     '#FF0000',
#     '#0000FF',
# ]

l_hex_str = ['00', '40', '80', 'C0', 'FF']
l_color = ['#{}{}{}'.format(col_r, col_g, col_b) for col_r in l_hex_str for col_g in l_hex_str for col_b in l_hex_str]

class CalcClusterData(Exception):
    __slot__ = [
        'cluster_points',
        'l_cluster_points_correspond',
        'arr_error',
        'l_error_cluster',
        'arr_argmin',
    ]

    def __init__(
        self,
        cluster_points: np.ndarray,
        l_cluster_points_correspond: List[np.ndarray],
        arr_error: np.ndarray,
        l_error_cluster: List[List[np.float128]],
        arr_argmin: np.ndarray
    ):
        self.cluster_points = cluster_points
        self.l_cluster_points_correspond = l_cluster_points_correspond
        self.arr_error = arr_error
        self.l_error_cluster = l_error_cluster
        self.arr_argmin = arr_argmin


def calculate_clusters(points: np.ndarray, cluster_amount: int, iterations: int, epsilon: float=0.0001) \
-> CalcClusterData:
    point_dim = points.shape[1]
    # cluster_amount <= points.shape[0] !!!
    cluster_points = points[np.random.permutation(np.arange(0, len(points)))[:cluster_amount]].copy()
    # print("before cluster_points:\n{}".format(cluster_points))

    # calc new clusters!
    l_error: List[np.float128] = []
    l_error_cluster: List[List[np.float128]] = [[] for _ in range(0, cluster_amount)]

    cluster_points_prev: np.ndarray = cluster_points.copy()
    i_nr: int
    for i_nr in range(0, iterations + 1):
        arr_sums_diff = np.sqrt(np.sum((points.reshape((-1, 1, point_dim)) - cluster_points.reshape((1, -1, point_dim)))**2, axis=2))

        arr_argmin = np.argmin(arr_sums_diff, axis=1)

        u, c = np.unique(arr_argmin, return_counts=True)
        assert c.shape[0] == cluster_amount

        # error = np.sum(arr_sums_diff)     

        cluster_points_prev[:] = cluster_points

        l_error_cluster_one = []

        i: int
        for i in range(0, cluster_amount):
            arr_idxs: np.ndarray = arr_argmin==i
            if arr_idxs.shape[0] == 0:
                continue
            arr = points[arr_idxs]
            error_cluster = np.mean(arr_sums_diff[arr_idxs, i])
            l_error_cluster_one.append(error_cluster)
            l_error_cluster[i].append(error_cluster)
            cluster_points[i] = np.mean(arr, axis=0)

        error = np.sum(l_error_cluster_one)
        print("i_nr: {}, error: {}".format(i_nr, error))
        l_error.append(error)

        if np.all(np.equal(cluster_points, cluster_points_prev)):
            break

        # print("- after cluster_points:\n{}".format(cluster_points))

    arr_error = np.array(l_error)

    arr_sums_diff = np.sqrt(np.sum((points.reshape((-1, 1, point_dim)) - cluster_points.reshape((1, -1, point_dim))) ** 2, axis=2))
    arr_argmin = np.argmin(arr_sums_diff, axis=1)
    l_cluster_points_correspond = []
    for i in range(0, cluster_amount):
        l_cluster_points_correspond.append(points[arr_argmin==i])

    return CalcClusterData(cluster_points, l_cluster_points_correspond, arr_error, l_error_cluster, arr_argmin)


def get_plots(cluster_points, l_cluster_points_correspond, arr_error, l_error_cluster):
    cluster_amount = cluster_points.shape[0]
    plt.close('all')

    if cluster_points.shape[1] == 2:
        plt.figure()

        for color, cluster_points_correspond in zip(l_color, l_cluster_points_correspond):
            xs_i, ys_i = cluster_points_correspond.T
            plt.plot(xs_i, ys_i, color=color, marker='.', ms=2., ls='')
        # plt.plot(xs, ys, color='#0000FF', marker='.', ms=2., ls='')
        xs_c, ys_c = cluster_points.T
        plt.plot(xs_c, ys_c, color='#00FF00', marker='.', ms=8., ls='', mec='#000000')

        plt.title('Cluster scatter plot')
        plt.tight_layout()


    plt.figure()

    plt.plot(np.arange(0, arr_error.shape[0]), arr_error, color='#00FF00', marker='.', ms=8., ls='-')

    plt.title('Error curve')
    plt.tight_layout()


    plt.figure()

    xs = np.arange(0, arr_error.shape[0])
    l_handler_names = []
    for i in range(0, cluster_amount):
        p = plt.plot(xs, l_error_cluster[i], color=l_color[i], marker='.', ms=8., ls='-')[0]
        l_handler_names.append((p, 'p{}'.format(i)))
    # p0 = plt.plot(xs, l_error_cluster[0], color=l_color[0], marker='.', ms=8., ls='-')[0]
    # p1 = plt.plot(xs, l_error_cluster[1], color=l_color[1], marker='.', ms=8., ls='-')[0]
    # p2 = plt.plot(xs, l_error_cluster[2], color=l_color[2], marker='.', ms=8., ls='-')[0]
    # p3 = plt.plot(xs, l_error_cluster[3], color=l_color[3], marker='.', ms=8., ls='-')[0]

    # l_legend = [(p0, p1, p2, p3), ('p0', 'p1', 'p2', 'p3')]
    plt.legend(*list(zip(*l_handler_names)))

    plt.title('Many error curves')
    plt.tight_layout()


    plt.show()


def do_clustering_silhouette(points, l_cluster, cluster_amount):
    l_points_in_cluster = [[] for _ in range(0, cluster_amount)]
    for point, c in zip(points, l_cluster):
        l_points_in_cluster[c].append(point)

    l_arr_points_in_cluster = [np.array(l) for l in l_points_in_cluster]

    l_cluster_val_a = [np.zeros((arr.shape[0], )) for arr in l_arr_points_in_cluster]
    l_cluster_val_b = [np.zeros((arr.shape[0], )) for arr in l_arr_points_in_cluster]
    l_cluster_val_s = [np.zeros((arr.shape[0], )) for arr in l_arr_points_in_cluster]

    for cluster_nr_i, (arr_a, arr_b, arr_s, arr_points_in_cluster_i) in enumerate(zip(
        l_cluster_val_a, l_cluster_val_b, l_cluster_val_s, l_arr_points_in_cluster
    ), 0):
        rows = arr_points_in_cluster_i.shape[0]
        
        if rows == 1:
            arr_a[0] = 0
            arr_b[0] = 0
            arr_s[0] = 0
            continue
        
        for i, p1 in enumerate(arr_points_in_cluster_i, 0):
            val_a = np.sum(np.sqrt(np.sum((arr_points_in_cluster_i - p1)**2, axis=1))) / (rows - 1)
            arr_a[i] = val_a
            
            l_b_vals = []
            for cluster_nr_j, arr_points_in_cluster_j in enumerate(l_arr_points_in_cluster, 0):
                if cluster_nr_j == cluster_nr_i:
                    continue
                
                val_b_i = np.sum(np.sqrt(np.sum((arr_points_in_cluster_j - p1)**2, axis=1))) / (rows - 1)
                l_b_vals.append(val_b_i)

            val_b = np.min(l_b_vals)
            arr_b[i] = val_b

            arr_s[i] = 1 - val_a / val_b if val_a < val_b else val_b / val_a - 1

    return DotMap(locals(), _dynamic=None)
