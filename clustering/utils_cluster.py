from typing import List, Dict, Set, Mapping, Any, Tuple

import matplotlib.pyplot as plt

from dotmap import DotMap
import numpy as np

__version__ = '0.1.0'
dm_obj_file_name = 'dm_obj.pkl.gz'

# l_color = [
#	 '#00F020',
#	 '#008000',
#	 '#FF0000',
#	 '#0000FF',
# ]

l_hex_str = ['00', '40', '80', 'C0', 'FF']
l_color = ['#{}{}{}'.format(col_r, col_g, col_b) for col_r in l_hex_str for col_g in l_hex_str for col_b in l_hex_str]

class CalcClusterData(Exception):

	__slot__ = [
		'cluster_amount',
		'arr_cluster_mean_vec',
		'l_cluster_points_correspond',
		'arr_error',
		'l_error_cluster',
		'arr_cluster_nr',
	]

	def __init__(
		self,
		cluster_amount: int,
		arr_cluster_mean_vec: np.ndarray,
		l_cluster_points_correspond: List[np.ndarray],
		arr_error: np.ndarray,
		l_error_cluster: List[List[np.float64]],
		arr_cluster_nr: np.ndarray,
	):
		self.cluster_amount = cluster_amount
		self.arr_cluster_mean_vec = arr_cluster_mean_vec
		self.l_cluster_points_correspond = l_cluster_points_correspond
		self.arr_error = arr_error
		self.l_error_cluster = l_error_cluster
		self.arr_cluster_nr = arr_cluster_nr


class CalcSilhouetteData(Exception):

	__slot__ = [
		'cluster_amount',
		'arr_point',
		'arr_cluster_nr',
		'l_arr_points_in_cluster',
		'l_cluster_val_a',
		'l_cluster_val_b',
		'l_cluster_val_s',
		'val_s_mean',
	]

	def __init__(
		self,
		cluster_amount: int,
		arr_point: np.ndarray,
		arr_cluster_nr: np.ndarray,
		l_arr_points_in_cluster: List[np.ndarray],
		l_cluster_val_a: List[np.ndarray],
		l_cluster_val_b: List[np.ndarray],
		l_cluster_val_s: List[np.ndarray],
		val_s_mean: np.float64,
	):
		self.cluster_amount = cluster_amount
		self.arr_point = arr_point
		self.arr_cluster_nr = arr_cluster_nr
		self.l_arr_points_in_cluster = l_arr_points_in_cluster
		self.l_cluster_val_a = l_cluster_val_a
		self.l_cluster_val_b = l_cluster_val_b
		self.l_cluster_val_s = l_cluster_val_s
		self.val_s_mean = val_s_mean


def calculate_clusters(
	cluster_amount: int,
	arr_point: np.ndarray,
	iterations: int,
	epsilon: float=0.0001,
	should_print_values: bool=False,
) -> CalcClusterData:
	point_dim = arr_point.shape[1]
	# cluster_amount <= arr_point.shape[0] !!!

	arr_cluster_mean_vec = arr_point[np.random.permutation(np.arange(0, len(arr_point)))[:cluster_amount]].copy()
	# print("before arr_cluster_mean_vec:\n{}".format(arr_cluster_mean_vec))

	# calc new clusters!
	l_error: List[np.float64] = []
	l_error_cluster: List[List[np.float64]] = [[] for _ in range(0, cluster_amount)]

	cluster_points_prev: np.ndarray = arr_cluster_mean_vec.copy()
	i_nr: int
	for i_nr in range(0, iterations + 1):
		arr_sums_diff = np.sqrt(np.sum((arr_point.reshape((-1, 1, point_dim)) - arr_cluster_mean_vec.reshape((1, -1, point_dim)))**2, axis=2))

		arr_cluster_nr = np.argmin(arr_sums_diff, axis=1)

		u, c = np.unique(arr_cluster_nr, return_counts=True)
		# print(f"i_nr: {i_nr}")
		# print(f"- u.shape: {u.shape}")
		# print(f"- c.shape: {c.shape}")
		assert c.shape[0] == cluster_amount

		# error = np.sum(arr_sums_diff)

		cluster_points_prev[:] = arr_cluster_mean_vec

		l_error_cluster_one = []

		i: int
		for i in range(0, cluster_amount):
			arr_idxs: np.ndarray = arr_cluster_nr==i
			if arr_idxs.shape[0] == 0:
				continue
			arr = arr_point[arr_idxs]
			error_cluster = np.mean(arr_sums_diff[arr_idxs, i])
			l_error_cluster_one.append(error_cluster)
			l_error_cluster[i].append(error_cluster)
			arr_cluster_mean_vec[i] = np.mean(arr, axis=0)

		error = np.sum(l_error_cluster_one)
		if should_print_values:
			print("i_nr: {}, error: {}".format(i_nr, error))
		l_error.append(error)

		if np.all(np.equal(arr_cluster_mean_vec, cluster_points_prev)):
			break

		# print("- after arr_cluster_mean_vec:\n{}".format(arr_cluster_mean_vec))

	arr_error = np.array(l_error)

	arr_sums_diff = np.sqrt(np.sum((arr_point.reshape((-1, 1, point_dim)) - arr_cluster_mean_vec.reshape((1, -1, point_dim))) ** 2, axis=2))
	arr_cluster_nr = np.argmin(arr_sums_diff, axis=1)
	l_cluster_points_correspond = []
	for i in range(0, cluster_amount):
		l_cluster_points_correspond.append(arr_point[arr_cluster_nr==i])

	return CalcClusterData(
		cluster_amount=cluster_amount,
		arr_cluster_mean_vec=arr_cluster_mean_vec,
		l_cluster_points_correspond=l_cluster_points_correspond,
		arr_error=arr_error,
		l_error_cluster=l_error_cluster,
		arr_cluster_nr=arr_cluster_nr,
	)


def calculate_clustering_silhouette(
	cluster_amount: int,
	arr_point: np.ndarray,
	arr_cluster_nr: np.ndarray,
) -> CalcSilhouetteData:
	l_points_in_cluster_nr = [[] for _ in range(0, cluster_amount)]
	for point, cluster_nr in zip(arr_point, arr_cluster_nr):
		l_points_in_cluster_nr[cluster_nr].append(point)

	assert all([len(l) > 0 for l in l_points_in_cluster_nr]) and "Every cluster_nr should contain at least one element!"

	l_arr_points_in_cluster = [np.array(l) for l in l_points_in_cluster_nr]

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

	val_s_mean = np.mean(np.hstack(l_cluster_val_s))

	calc_silouhette_data = CalcSilhouetteData(
		cluster_amount=cluster_amount,
		arr_point=arr_point,
		arr_cluster_nr=arr_cluster_nr,
		l_arr_points_in_cluster=l_arr_points_in_cluster,
		l_cluster_val_a=l_cluster_val_a,
		l_cluster_val_b=l_cluster_val_b,
		l_cluster_val_s=l_cluster_val_s,
		val_s_mean=val_s_mean,
	)

	return calc_silouhette_data


def calculate_cluster_points_and_silouhette(
	arr_point: np.ndarray,
	min_amount_cluster: int,
	max_amount_cluster: int,
	iterations: int,
) -> List[Dict[str, object]]:
	l_d_data = []
	for cluster_amount in range(min_amount_cluster, max_amount_cluster+1):
		print(f"cluster_amount: {cluster_amount}")

		assert len(l_color) >= cluster_amount

		calc_cluster_data = calculate_clusters(
			cluster_amount=cluster_amount,
			arr_point=arr_point,
			iterations=iterations,
		)

		calc_silouhette_data =  calculate_clustering_silhouette(
			cluster_amount=cluster_amount,
			arr_point=arr_point,
			arr_cluster_nr=calc_cluster_data.arr_cluster_nr,
		)
		val_s_mean = calc_silouhette_data.val_s_mean

		print(f"(cluster_amount, val_s_mean): {(cluster_amount, val_s_mean)}")

		l_d_data.append({
			'cluster_amount': cluster_amount,
			'calc_cluster_data': calc_cluster_data,
			'calc_silouhette_data': calc_silouhette_data,
		})

	return l_d_data


def find_best_fitting_cluster_amount(
	max_try_nr: int,
	arr_point: np.ndarray,
	min_amount_cluster: int,
	max_amount_cluster: int,
	iterations: int,
) -> np.ndarray:
	l_try_nr_arr_x_cluster_amount_arr_y_s_mean = []
		
	for try_nr in range(1, max_try_nr+1):
		l_d_data = calculate_cluster_points_and_silouhette(
			arr_point=arr_point,
			min_amount_cluster=2,
			max_amount_cluster=15,
			iterations=iterations,
		)

		l_cluster_amount_s_mean = [(d['cluster_amount'], d['calc_silouhette_data'].val_s_mean) for d in l_d_data]

		arr_x_cluster_amount, arr_y_s_mean = [np.array(list(l)) for l in zip(*l_cluster_amount_s_mean)]
		l_try_nr_arr_x_cluster_amount_arr_y_s_mean.append((try_nr, arr_x_cluster_amount, arr_y_s_mean))

	arr_arr_y_s_mean = np.vstack([arr_y_s_mean for _, _, arr_y_s_mean in l_try_nr_arr_x_cluster_amount_arr_y_s_mean])

	arr_arr_y_s_mean_argsort_inv = np.argsort(arr_arr_y_s_mean, axis=1)
	arr_arr_y_s_mean_argsort = arr_arr_y_s_mean_argsort_inv.copy()
	arr_arange = np.arange(0, arr_arr_y_s_mean.shape[1])
	for i in range(0, arr_arr_y_s_mean.shape[0]):
		arr_arr_y_s_mean_argsort[i, arr_arr_y_s_mean_argsort_inv[i]] = arr_arange

	arr_x_cluster_amount = l_try_nr_arr_x_cluster_amount_arr_y_s_mean[0][1]
	arr_best_cluster_amount = arr_x_cluster_amount[np.argsort(np.sum(arr_arr_y_s_mean_argsort, axis=0))[::-1]]

	print(f"arr_best_cluster_amount: {arr_best_cluster_amount}")

	plt.figure()

	plt.title("l_cluster_amount_s_mean")

	l_p = []
	for try_nr, arr_x_cluster_amount, arr_y_s_mean in l_try_nr_arr_x_cluster_amount_arr_y_s_mean:
		p = plt.plot(arr_x_cluster_amount, arr_y_s_mean, linestyle='-', marker='o', label=f"try_nr: {try_nr}")[0]
		l_p.append(p)

	plt.legend()

	plt.show(block=True)

	return arr_best_cluster_amount


def get_plots(arr_cluster_mean_vec, l_cluster_points_correspond, arr_error, l_error_cluster):
	cluster_amount = arr_cluster_mean_vec.shape[0]
	plt.close('all')

	if arr_cluster_mean_vec.shape[1] == 2:
		plt.figure()

		for color, cluster_points_correspond in zip(l_color, l_cluster_points_correspond):
			xs_i, ys_i = cluster_points_correspond.T
			plt.plot(xs_i, ys_i, color=color, marker='.', ms=2., ls='')
		# plt.plot(xs, ys, color='#0000FF', marker='.', ms=2., ls='')
		xs_c, ys_c = arr_cluster_mean_vec.T
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
