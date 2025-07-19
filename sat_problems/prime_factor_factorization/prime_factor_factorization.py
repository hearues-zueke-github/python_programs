#! /usr/bin/python3.12

import sys

import numpy as np

from pysat.formula import CNF
from pysat.solvers import Solver

def cnf_logic_equal(a, b):
	return [(a, -b), (-a, b)]


def cnf_logic_not(a, b):
	return [(a, b), (-a, -b)]


def cnf_logic_and(a, b, c):
	return [(-a, -b, c), (a, -c), (b, -c)]


def cnf_logic_or(a, b, c):
	return [(a, b, -c), (-a, c), (-b, c)]


def cnf_logic_xor(a, b, c):
	return [(-a, -b, -c), (a, b, -c), (-a, b, c), (a, -b, c)]


def cnf_logic_and_many(l_a, b):
	return [tuple(-a for a in l_a) + (b, )] + [(a, -b) for a in l_a]


def cnf_logic_or_many(l_a, b):
	return [tuple(a for a in l_a) + (-b, )] + [(-a, b) for a in l_a]


def cnf_logic_adder_1_bit(a, b, c, r, s, l_temp):
	assert len(l_temp) == 7

	cnf = []
	
	cnf.extend(cnf_logic_and_many([a, b, c], l_temp[0]))

	cnf.extend(cnf_logic_and_many([a, -b, -c], l_temp[1]))
	cnf.extend(cnf_logic_and_many([-a, -b, c], l_temp[2]))
	cnf.extend(cnf_logic_and_many([-a, b, -c], l_temp[3]))

	cnf.extend(cnf_logic_and_many([a, b, -c], l_temp[4]))
	cnf.extend(cnf_logic_and_many([a, -b, c], l_temp[5]))
	cnf.extend(cnf_logic_and_many([-a, b, c], l_temp[6]))

	cnf.extend(cnf_logic_or_many([l_temp[0], l_temp[1], l_temp[2], l_temp[3]], s))
	cnf.extend(cnf_logic_or_many([l_temp[0], l_temp[4], l_temp[5], l_temp[6]], r))

	return cnf


"""
	l_a[0] is bit 1, l_a[len(l_a) - 1] is last bit
	l_b[0] is bit 1, l_b[len(l_b) - 1] is last bit
	also for l_carry and l_result
"""
def cnf_logic_adder(l_a, l_b, l_result, l_temp_all):
	length = len(l_a)

	assert length == len(l_b)
	assert len(l_result) == length + 1
	assert len(l_temp_all) == (length - 1) * 7 + length

	l_carry = l_temp_all[:length]
	l_temp = l_temp_all[length:]

	# do step by step for each bit
	cnf = []

	cnf.extend(cnf_logic_and(l_a[0], l_b[0], l_carry[0]))
	cnf.extend(cnf_logic_xor(l_a[0], l_b[0], l_result[0]))

	for i in range(1, length):
		cnf.extend(cnf_logic_adder_1_bit(l_a[i], l_b[i], l_carry[i - 1], l_carry[i], l_result[i], l_temp[(i-1)*7 : i*7]))

	cnf.extend(cnf_logic_equal(l_carry[length - 1], l_result[length]))

	return cnf


def cnf_logic_multiplier(l_a, l_b, l_result, l_temp_all):
	length_a = len(l_a)
	length_b = len(l_b)
	assert length_a >= 1
	assert length_b >= 1
	assert length_a >= length_b

	length_logic_and_temp = length_a * length_b
	length_temp_addition_carry = 7 * (length_a - 1) + length_a

	if length_b == 1:
		assert len(l_temp_all) == 0
		assert length_a == len(l_result)

		cnf = []

		for i_a in range(0, length_a):
			cnf.extend(cnf_logic_and(a=l_a[i_a], b=l_b[0], c=l_result[i_a]))

		return cnf

	assert length_b >= 2
	assert length_a + length_b == len(l_result)

	length_temp_all = (
		length_logic_and_temp +								# for all and connections between all a_i_j and b_k_l variables
		1 +												# temp variable needed for bit 0 in the addition part (for the first addition)
		length_temp_addition_carry * (length_b - 1) +	# needed temp variables for addition only for carry and the addition part
		length_a * (length_b - 2)						# the needed result temp variables, except for the last part and the first bit
	)

	assert len(l_temp_all) == length_temp_all

	iter_num = 0

	temp_var_bit_zero = l_temp_all[iter_num]
	iter_num += 1

	l_a_and_b = [l_temp_all[iter_num+i*length_a:iter_num+(i+1)*length_a] for i in range(0, length_b)]
	iter_num += length_a * length_b

	l_l_temp_rest = [l_temp_all[iter_num+i*(7*(length_a-1)+length_a):iter_num+(i+1)*(7*(length_a-1)+length_a)] for i in range(0, length_b - 1)]
	iter_num += (7 * (length_a - 1) + length_a) * (length_b - 1)

	l_l_temp_result = [l_temp_all[iter_num+i*length_a:iter_num+(i+1)*length_a] for i in range(0, length_b - 2)]
	iter_num += length_a * (length_b - 2)

	assert length_temp_all == iter_num

	cnf = []

	globals()['d'] = locals()

	# first do logical 'and' combination of 'a' and 'b' values
	for i_a in range(0, length_a):
		for i_b in range(0, length_b):
			cnf.extend(cnf_logic_and(a=l_a[i_a], b=l_b[i_b], c=l_a_and_b[i_b][i_a]))

	# next set the last logical 'and' to the result equal
	cnf.extend(cnf_logic_equal(a=l_a_and_b[0][0], b=l_result[0]))

	cnf.append((-temp_var_bit_zero, ))

	if length_b >= 3:
		# next do the first addition part
		cnf.extend(cnf_logic_adder(l_a=l_a_and_b[0][1:]+[temp_var_bit_zero], l_b=l_a_and_b[1], l_result=[l_result[1]]+l_l_temp_result[0], l_temp_all=l_l_temp_rest[0]))
		
		# do the next inner parts
		for i in range(1, length_b - 2):
			cnf.extend(cnf_logic_adder(l_a=l_a_and_b[i + 1], l_b=l_l_temp_result[i - 1], l_result=[l_result[i + 1]]+l_l_temp_result[i], l_temp_all=l_l_temp_rest[i]))

		# and finally the last addition layer
		cnf.extend(cnf_logic_adder(l_a=l_a_and_b[length_b - 1], l_b=l_l_temp_result[length_b - 3], l_result=l_result[length_b - 1:], l_temp_all=l_l_temp_rest[length_b - 2]))
	else:
		cnf.extend(cnf_logic_adder(l_a=l_a_and_b[0][1:]+[temp_var_bit_zero], l_b=l_a_and_b[1], l_result=l_result[1:], l_temp_all=l_l_temp_rest[0]))


	return cnf


if __name__ == '__main__':
	# test if a number p is able to be split up in two numbers
	bits_a = 5
	bits_b = bits_a

	bits_sum = bits_a + 1

	num_var = 1

	assert bits_a >= bits_b

	l_a = [num_var + i for i in range(0, bits_a)]
	num_var += bits_a

	l_b = [num_var + i for i in range(0, bits_b)]
	num_var += bits_b

	l_result = [num_var + i for i in range(0, bits_sum)]
	num_var += bits_a + bits_b

	amount_temp = 7 * (bits_a - 1) + bits_a
	
	l_temp_all = [num_var + i for i in range(0, amount_temp)]
	num_var += amount_temp

	# cnf = cnf_logic_multiplier(l_a=l_a, l_b=l_b, l_result=l_result, l_temp_all=l_temp_all)
	cnf = cnf_logic_adder(l_a=l_a, l_b=l_b, l_result=l_result, l_temp_all=l_temp_all)

	num_sum_max = 2**(bits_a+1)

	for num_sum in range(0, num_sum_max):
		str_sum = bin(num_sum)[2:]
		if len(str_sum) < bits_sum:
			str_sum = '0' * (bits_sum - len(str_sum)) + str_sum
		l_bit_sum = list(map(int, str_sum[::-1]))

		print(f'str_sum: {str_sum}, num_sum: {num_sum}')

		for result, bit_sum in zip(l_result, l_bit_sum):
			if bit_sum == 0:
				cnf.append((-result, ))
			else:
				cnf.append((result, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		cnf = cnf[:-bits_sum]

		l_num_a_b = []

		for i, model in enumerate(l_model, 0):
			l_bit_a = [1 if model[a - 1] > 0 else 0 for a in l_a]
			l_bit_b = [1 if model[a - 1] > 0 else 0 for a in l_b]

			num_a = np.sum(np.array(l_bit_a) * 2**np.arange(0, bits_a))
			num_b = np.sum(np.array(l_bit_b) * 2**np.arange(0, bits_b))

			str_a = ''.join(map(str, l_bit_a[::-1]))
			str_b = ''.join(map(str, l_bit_b[::-1]))

			l_num_a_b.append((num_a, num_b))

		print(f'l_num_a_b: {l_num_a_b}')
		print(f'len(l_num_a_b): {len(l_num_a_b)}')

	sys.exit()


	# test if a number p is able to be split up in two numbers
	bits_a = 20
	bits_b = bits_a

	bits_p = bits_a + bits_b

	num_var = 1

	assert bits_a >= bits_b

	l_a = [num_var + i for i in range(0, bits_a)]
	num_var += bits_a

	l_b = [num_var + i for i in range(0, bits_b)]
	num_var += bits_b

	# l_result = [num_var + i for i in range(0, bits_a)]
	l_result = [num_var + i for i in range(0, bits_p)]
	num_var += bits_a + bits_b

	if bits_b == 1:
		amount_temp = 0
	else:
		amount_temp = bits_a * bits_b + 1 + (7 * (bits_a - 1) + bits_a) * (bits_b - 1) + bits_a * (bits_b - 2)
	
	l_temp_all = [num_var + i for i in range(0, amount_temp)]
	num_var += amount_temp

	# cnf = cnf_logic_multiplier(l_a=l_a, l_b=l_b, l_result=l_result, l_temp_all=l_temp_all)
	cnf = cnf_logic_multiplier(l_a=l_a, l_b=l_b, l_result=l_result, l_temp_all=l_temp_all)

	num_p_max = 2**bits_a

	# for num_p in range(1, num_p_max):
	for num_p in range(num_p_max - 10, num_p_max):
		str_p = bin(num_p)[2:]
		if len(str_p) < bits_p:
			str_p = '0' * (bits_p - len(str_p)) + str_p
		l_bit_p = list(map(int, str_p[::-1]))

		print(f'str_p: {str_p}, num_p: {num_p}')

		for result, bit_p in zip(l_result, l_bit_p):
			if bit_p == 0:
				cnf.append((-result, ))
			else:
				cnf.append((result, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		cnf = cnf[:-bits_p]

		l_num_a_b = []

		for i, model in enumerate(l_model, 0):
			l_bit_a = [1 if model[a - 1] > 0 else 0 for a in l_a]
			l_bit_b = [1 if model[a - 1] > 0 else 0 for a in l_b]

			num_a = np.sum(np.array(l_bit_a) * 2**np.arange(0, bits_a))
			num_b = np.sum(np.array(l_bit_b) * 2**np.arange(0, bits_b))

			str_a = ''.join(map(str, l_bit_a[::-1]))
			str_b = ''.join(map(str, l_bit_b[::-1]))

			l_num_a_b.append((num_a, num_b))

		print(f'l_num_a_b: {l_num_a_b}')

	sys.exit()

	for bits_b in range(2, 6):
		for bits_a in range(bits_b, bits_b + 4):
			print(f"bits_a: {bits_a}, bits_b: {bits_b}")

			num_var = 1

			assert bits_a >= bits_b

			l_a = [num_var + i for i in range(0, bits_a)]
			num_var += bits_a

			l_b = [num_var + i for i in range(0, bits_b)]
			num_var += bits_b

			# l_result = [num_var + i for i in range(0, bits_a)]
			l_result = [num_var + i for i in range(0, bits_a + bits_b)]
			num_var += bits_a + bits_b

			if bits_b == 1:
				amount_temp = 0
			else:
				amount_temp = bits_a * bits_b + 1 + (7 * (bits_a - 1) + bits_a) * (bits_b - 1) + bits_a * (bits_b - 2)
			
			l_temp_all = [num_var +i for i in range(0, amount_temp)]
			num_var += amount_temp

			# cnf = cnf_logic_multiplier(l_a=l_a, l_b=l_b, l_result=l_result, l_temp_all=l_temp_all)
			cnf = cnf_logic_multiplier(l_a=l_a, l_b=l_b, l_result=l_result, l_temp_all=l_temp_all)


			for num_a in range(0, 2**bits_a):
				str_a = bin(num_a)[2:]
				if len(str_a) < bits_a:
					str_a = '0' * (bits_a - len(str_a)) + str_a
				l_bit_a = list(map(int, str_a[::-1]))

				for num_b in range(0, 2**bits_b):
					str_b = bin(num_b)[2:]
					if len(str_b) < bits_b:
						str_b = '0' * (bits_b - len(str_b)) + str_b
					l_bit_b = list(map(int, str_b[::-1]))

					num_p = num_a * num_b
					l_bit_p = list(map(int, bin(num_p)[2:][::-1]))

					bits_p = bits_a + bits_b
					if len(l_bit_p) < bits_p:
						l_bit_p += [0] * (bits_p - len(l_bit_p))

					str_p = ''.join(map(str, l_bit_p[::-1]))

					for a, bit_a in zip(l_a, l_bit_a):
						if bit_a == 0:
							cnf.append((-a, ))
						else:
							cnf.append((a, ))

					for b, bit_b in zip(l_b, l_bit_b):
						if bit_b == 0:
							cnf.append((-b, ))
						else:
							cnf.append((b, ))

					l_model = []
					with Solver(bootstrap_with=cnf) as solver:
						is_solveable = solver.solve()
						
						for  model in solver.enum_models():
							l_model.append(model)

					assert len(l_model) == 1

					model = l_model[0]
					l_bit_p_sat = [1 if model[result-1] > 0 else 0 for result in l_result]

					str_p_sat = ''.join(map(str, l_bit_p_sat[::-1]))
					num_p_sat = np.sum(l_bit_p_sat * 2**np.arange(0, bits_p))

					assert num_p == num_p_sat

					cnf = cnf[:-bits_p]

	sys.exit()

	# make first a 4 bit multiplier, later can be expanded to bigger one

	# define the two 4 bit numbers as 'a' and 'b'
	
	glob_var_counter = 1

	bits = 4
	d_var_to_int = {}
	
	for i in range(0, bits):
		d_var_to_int[f'a_{i}'] = glob_var_counter
		glob_var_counter += 1

	for i in range(0, bits):
		d_var_to_int[f'b_{i}'] = glob_var_counter
		glob_var_counter += 1

	cnf = []

	# prepare the 'and bit' operations of each row
	for iter_a in range(0, bits):
		for iter_b in range(0, bits):
			i_v = glob_var_counter
			glob_var_counter += 1

			d_var_to_int[f'v_{iter_a}_{iter_b}'] = i_v

			i_a = d_var_to_int[f'a_{iter_a}']
			i_b = d_var_to_int[f'b_{iter_b}']

			cnf.extend(cnf_logic_or(i_a, i_b, i_v))

	# now do two rows sum, starting from the bottom row, and go up
	
	# first lets do the last two rows
	d_var_to_int['p_0'] = d_var_to_int[f'v_{bits-1}_0']



	print(f'd_var_to_int: {d_var_to_int}')

	cnf = CNF(from_clauses=[[-1, 2], [1, -2]])

	with Solver(bootstrap_with=cnf) as solver:
		print('formula is', f'{"s" if solver.solve() else "uns"}atisfiable')

		for i, model in enumerate(solver.enum_models()):
			print(f"i: {i}, model: {model}")
