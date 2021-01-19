from typing import List, Dict, Set, Mapping, Any, Tuple

import numpy as np

dm_obj_file_name = 'dm_obj.pkl.gz'


def calculate_clusters(points : np.ndarray, cluster_amount : int, iterations : int) -> Tuple[np.ndarray, np.ndarray]:
    point_dim = points.shape[1]
    # cluster_amount <= points.shape[0] !!!
    cluster_points = points[np.random.permutation(np.arange(0, len(points)))[:cluster_amount]].copy()
    print("before cluster_points:\n{}".format(cluster_points))

    # calc new clusters!
    l_error : List[float] = []

    i_nr : int
    for i_nr in range(0, iterations + 1):
        arr_sums_diff = np.sqrt(np.sum((points.reshape((-1, 1, point_dim)) - cluster_points.reshape((1, -1, point_dim)))**2, axis=2))

        arr_argmin = np.argmin(arr_sums_diff, axis=1)
        
        # error = np.sum(arr_sums_diff)
        error = np.sum(arr_sums_diff[:, arr_argmin])
        l_error.append(error)
        print("i_nr: {}, error: {}".format(i_nr, error))
        
        i : int
        for i in range(0, cluster_amount):
            arr = points[arr_argmin==i]
            cluster_points[i] = np.mean(arr, axis=0)
        
        # print("- after cluster_points:\n{}".format(cluster_points))

    arr_error = np.array(l_error)
    
    return cluster_points, arr_error
