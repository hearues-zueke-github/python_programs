import numpy as np

def increment_arr_uint8(arr: np.ndarray) -> None:
	for i in range(0, arr.shape[0]):
		v = arr[i]
		arr[i] += 1
		if v < np.uint8(0xFF):
			break
