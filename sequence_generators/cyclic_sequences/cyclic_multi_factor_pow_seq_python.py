import numpy as np

def calc_2_value_sequence_pow_2(
	modulo: int,
	factors: np.ndarray,
	values: np.ndarray,
):
	assert factors.shape[0] == 9
	
	cycle_len = modulo**2
	assert values.shape[0] == cycle_len*2
	
	values[0] = 0
	values[1] = 0

	cycle_checker = np.zeros((cycle_len, ), dtype=np.int8)

	x0 = 0
	x1 = 0
	for i in range(0, cycle_len):
		values[i*2+0] = x0
		values[i*2+1] = x1

		next_x = (
			(x0*x0 * x1*x1 * factors[0]) % modulo +

			(x0 * x1*x1 * factors[1]) % modulo +
			(x0*x0 * x1 * factors[2]) % modulo +

			(1 * x1*x1 * factors[3]) % modulo +
			(x0*x0 * 1 * factors[4]) % modulo +
			

			(x0 * x1 * factors[5]) % modulo +

			(1 * x1 * factors[6]) % modulo +
			(x0 * 1 * factors[7]) % modulo +


			(1 * 1 * factors[8])
		) % modulo

		x0 = x1
		x1 = next_x

		idx = x0 * modulo + x1
		if cycle_checker[idx] == 1:
			return 0

		cycle_checker[idx] = 1
		
	return 1


def calc_2_value_sequence_pow_2_many(
	modulo: int,
	amount_factors: int,
	factors: np.ndarray,
	values: np.ndarray,
	found_cycles: np.ndarray,
):
	assert factors.shape[0] == 9 * amount_factors
	
	cycle_len = modulo**2
	assert values.shape[0] == cycle_len*2 * amount_factors
	assert found_cycles.shape[0] == amount_factors

	cycle_checker = np.zeros((cycle_len, ), dtype=np.int8)

	for j in range(0, amount_factors):
		idx_val = j * cycle_len*2
		idx_fac = j * 9
		x0 = 0
		x1 = 0
		found_one_cycle = True
		for i in range(0, cycle_len):
			cycle_checker[i] = 0

		for i in range(0, cycle_len):
			values[idx_val + i*2+0] = x0
			values[idx_val + i*2+1] = x1

			next_x = (
				(x0**2 * x1**2 * factors[idx_fac + 0]) % modulo +

				(x0 * x1**2 * factors[idx_fac + 1]) % modulo +
				(x0**2 * x1 * factors[idx_fac + 2]) % modulo +

				(1 * x1**2 * factors[idx_fac + 3]) % modulo +
				(x0**2 * 1 * factors[idx_fac + 4]) % modulo +
				

				(x0 * x1 * factors[idx_fac + 5]) % modulo +

				(1 * x1 * factors[idx_fac + 6]) % modulo +
				(x0 * 1 * factors[idx_fac + 7]) % modulo +


				(1 * 1 * factors[idx_fac + 8])
			) % modulo

			x0 = x1
			x1 = next_x

			idx = x0 * modulo + x1
			if cycle_checker[idx] == 1:
				found_one_cycle = False
				break

			cycle_checker[idx] = 1

		found_cycles[j] = 1 if found_one_cycle else 0
