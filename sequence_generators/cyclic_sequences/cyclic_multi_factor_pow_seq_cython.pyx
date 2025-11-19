# cython: language_level=3
# distutils: language=c++

# solution.pyx

cimport numpy as np

from libcpp.vector cimport vector 

cpdef int calc_2_value_sequence_pow_1(
	long modulo,
	np.ndarray[np.int64_t, ndim=1] factors,
	np.ndarray[np.int64_t, ndim=1] values,
):
	assert factors.shape[0] == 4
	
	cdef long cycle_len = modulo**2
	assert values.shape[0] == cycle_len
	
	values[0] = 0
	values[1] = 0

	cdef vector[long] cycle_checker
	cycle_checker.reserve(cycle_len)

	for _ in range(0, cycle_len):
		cycle_checker.push_back(0)

	cdef long x0 = 0
	cdef long x1 = 0
	cdef long next_x
	cdef long idx
	for i in range(0, cycle_len):
		next_x = (
			(x0 * x1 * factors[0]) % modulo +
			(x0 * factors[1]) % modulo +
			(x1 * factors[2]) % modulo +
			factors[3]
		) % modulo

		idx = x0 * modulo + next_x
		if cycle_checker[idx] == 1:
			return 0

		cycle_checker[idx] = 1

		values[(i+1)*2+0] = x1
		values[(i+1)*2+1] = next_x

		x0 = x1
		x1 = next_x

	return 1


cpdef int calc_2_value_sequence_pow_2(
	long modulo,
	np.ndarray[np.int64_t, ndim=1] factors,
	np.ndarray[np.int64_t, ndim=1] values,
):
	assert factors.shape[0] == 9
	
	cdef long cycle_len = modulo**2
	assert values.shape[0] == cycle_len*2

	cdef vector[long] cycle_checker
	cycle_checker.reserve(cycle_len)

	for _ in range(0, cycle_len):
		cycle_checker.push_back(0)

	cdef long x0 = 0
	cdef long x1 = 0
	cdef long next_x
	cdef long idx
	for i in range(0, cycle_len):
		values[i*2+0] = x0
		values[i*2+1] = x1

		next_x = (
			(x0**2 * x1**2 * factors[0]) % modulo +

			(x0 * x1**2 * factors[1]) % modulo +
			(x0**2 * x1 * factors[2]) % modulo +

			(1 * x1**2 * factors[3]) % modulo +
			(x0**2 * 1 * factors[4]) % modulo +
			

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


cpdef void calc_2_value_sequence_pow_2_many(
	const long modulo,
	const long amount_factors,
	np.ndarray[np.int64_t, ndim=1] factors,
	np.ndarray[np.int64_t, ndim=1] values,
	np.ndarray[np.int64_t, ndim=1] found_cycles,
):
	assert factors.shape[0] == 9 * amount_factors
	
	cdef long cycle_len = modulo**2
	assert values.shape[0] == cycle_len*2 * amount_factors
	assert found_cycles.shape[0] == amount_factors

	cdef vector[long] cycle_checker
	cycle_checker.reserve(cycle_len)

	for _ in range(0, cycle_len):
		cycle_checker.push_back(0)

	cdef long x0
	cdef long x1
	cdef long next_x
	cdef long idx
	cdef bint found_one_cycle
	cdef long idx_val
	cdef long idx_fac
	
	cdef long fac_0
	cdef long fac_1
	cdef long fac_2
	cdef long fac_3
	cdef long fac_4
	cdef long fac_5
	cdef long fac_6
	cdef long fac_7
	cdef long fac_8

	for j in range(0, amount_factors):
		idx_val = j * cycle_len*2
		idx_fac = j * 9
		x0 = 0
		x1 = 0
		found_one_cycle = True
		for i in range(0, cycle_len):
			cycle_checker[i] = 0

		fac_0 = factors[idx_fac + 0]
		fac_1 = factors[idx_fac + 1]
		fac_2 = factors[idx_fac + 2]
		fac_3 = factors[idx_fac + 3]
		fac_4 = factors[idx_fac + 4]
		fac_5 = factors[idx_fac + 5]
		fac_6 = factors[idx_fac + 6]
		fac_7 = factors[idx_fac + 7]
		fac_8 = factors[idx_fac + 8]
		
		for i in range(0, cycle_len):
			values[idx_val + i*2+0] = x0
			values[idx_val + i*2+1] = x1

			next_x = (
				(x0**2 * x1**2 * fac_0) % modulo +

				(x0 * x1**2 * fac_1) % modulo +
				(x0**2 * x1 * fac_2) % modulo +

				(1 * x1**2 * fac_3) % modulo +
				(x0**2 * 1 * fac_4) % modulo +
				
				(x0 * x1 * fac_5) % modulo +

				(1 * x1 * fac_6) % modulo +
				(x0 * 1 * fac_7) % modulo +


				(1 * 1 * fac_8)
			) % modulo

			x0 = x1
			x1 = next_x

			idx = x0 * modulo + x1
			if cycle_checker[idx] == 1:
				found_one_cycle = False
				break

			cycle_checker[idx] = 1

		found_cycles[j] = 1 if found_one_cycle else 0
