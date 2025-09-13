#! /usr/bin/python2.7

import sys

import numpy as np

from datetime import datetime

def get_needed_seconds_for_function(func, args, kwargs):
	dt_start = datetime.now()
	
	ret_val = func(*args, **kwargs)

	dt_end = datetime.now()
	dt_diff = dt_end - dt_start

	return (dt_diff.total_seconds(), ret_val)


def get_primes(n):
	if n < 2:
		return
	yield 2
	if n < 3:
		return
	yield 3
	if n < 5:
		return
	yield 5
	ps = [2, 3, 5]
	d = [4, 2]

	i = 7
	j = 0
	while i <= n:
		q = int(np.sqrt(i))+1

		is_prime = True

		k = 2 # start from the prime number 5
		v = ps[k]
		while v < q:
			if i % v == 0:
				is_prime = False
				break
			k += 1
			v = ps[k]

		if is_prime:
			yield i
			ps.append(i)

		i += d[j]
		j = (j+1) % 2


def get_primes_list(n_max):
	if n_max < 2:
		return []

	if n_max < 3:
		return [2]

	if n_max < 5:
		return [2, 3]
	
	l_prime = [2, 3, 5]
	l_jump = [4, 2]
	jump_index = 0

	next_p = 7
	i_q = 1
	q = 3
	q_pow_2 = q**2
	while next_p <= n_max:
		is_prime = True

		for v in l_prime[2:i_q+1]: # start from the prime number 5
			if next_p % v == 0:
				is_prime = False
				break

		if is_prime:
			l_prime.append(next_p)

		next_p += l_jump[jump_index]

		if next_p >= q_pow_2:
			i_q += 1
			q = l_prime[i_q]
			q_pow_2 = q**2

		jump_index = (jump_index+1) % 2

	return l_prime


def get_primes_list_part(l_prime, n_min, n_max):
	last_prime = l_prime[len(l_prime) - 1]
	assert last_prime**2 >= n_max

	# assumption: l_prime is a sorted list of primes starting from 2, 3, 5 etc.

	# find the next starting possible prime
	# modulo 6 = 1, 7, 13, 19, etc. jump 4
	# modulo 6 = 5, 11, 17, 23, etc. jump 2
	next_p = n_min

	l_prime_next = []
	l_jump = [4, 2]
	jump_index = 0
	if next_p % 6 == 0:
		next_p += 1
	elif next_p % 6 >= 2:
		next_p += 5 - (next_p % 6)
		jump_index = 1

	for i_q in range(0, len(l_prime)):
		q = l_prime[i_q]
		q_pow_2 = q**2
		if q_pow_2 > next_p:
			break

	while next_p <= n_max:
		is_prime = True

		for v in l_prime[2:i_q+1]: # start from the prime number 5
			if next_p % v == 0:
				is_prime = False
				break

		if is_prime:
			l_prime_next.append(next_p)

		next_p += l_jump[jump_index]

		if next_p >= q_pow_2:
			i_q += 1
			q = l_prime[i_q]
			q_pow_2 = q**2

		jump_index = (jump_index+1) % 2

	return l_prime_next


def test_get_primes_list_part():
	l_prime = get_primes_list(n_max=100)

	l_prime_next = get_primes_list_part(l_prime=l_prime, n_min=101, n_max=5000)

	globals()['loc'] = locals()


def test_get_primes_list():
	d_n_max_to_l_prime = {
		0: [],
		1: [],
		2: [2],
		3: [2, 3],
		4: [2, 3],
		5: [2, 3, 5],
		10: [2, 3, 5, 7],
		100: [
			 2,  3,  5,  7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
			43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
		],
	}

	for n_max in sorted(d_n_max_to_l_prime.keys()):
		needed_seconds, ret_val = get_needed_seconds_for_function(func=get_primes_list, args=(), kwargs={'n_max': n_max})
		print(f'n_max: {n_max}, needed_seconds: {needed_seconds}')
		assert d_n_max_to_l_prime[n_max] == ret_val

	n_max_1 = 10000
	needed_seconds, l_1 = get_needed_seconds_for_function(func=get_primes_list, args=(), kwargs={'n_max': n_max_1})
	print(f'n_max: {n_max_1}, needed_seconds: {needed_seconds}')
	assert len(l_1) == 1229
	assert l_1[10] == 31
	assert l_1[100] == 547
	assert l_1[1000] == 7927
	assert l_1[len(l_1) - 1] == 9973

	n_max_2 = 100000
	needed_seconds, l_2 = get_needed_seconds_for_function(func=get_primes_list, args=(), kwargs={'n_max': n_max_2})
	print(f'n_max: {n_max_2}, needed_seconds: {needed_seconds}')
	assert len(l_2) == 9592
	assert l_2[:len(l_1)] == l_1
	assert l_2[10] == 31
	assert l_2[100] == 547
	assert l_2[1000] == 7927
	assert l_2[5000] == 48619
	assert l_2[len(l_2) - 1] == 99991

	n_max_3 = 1000000
	needed_seconds, l_3 = get_needed_seconds_for_function(func=get_primes_list, args=(), kwargs={'n_max': n_max_3})
	print(f'n_max: {n_max_3}, needed_seconds: {needed_seconds}')
	assert len(l_3) == 78498
	assert l_3[:len(l_2)] == l_2
	assert l_3[10] == 31
	assert l_3[100] == 547
	assert l_3[1000] == 7927
	assert l_3[10000] == 104743
	assert l_3[len(l_3) - 1] == 999983

	n_max_4 = 10000000
	needed_seconds, l_4 = get_needed_seconds_for_function(func=get_primes_list, args=(), kwargs={'n_max': n_max_4})
	print(f'n_max: {n_max_4}, needed_seconds: {needed_seconds}')
	assert l_4[:len(l_3)] == l_3


def test_different_prime_generating_functions():
	l_prime_1 = [v for v in get_primes(n=1000)]
	l_prime_2 = get_primes_list(n_max=1000)
	assert l_prime_1 == l_prime_2

	l_prime_small = get_primes_list(n_max=150)
	l_prime_3_part = get_primes_list_part(l_prime=l_prime_small, n_min=151, n_max=1000)
	l_prime_3 = l_prime_small + l_prime_3_part
	assert l_prime_3 == l_prime_1
	assert l_prime_3 == l_prime_2


def prime_factorization(n, ps):
	d = {}

	for p in ps:
		count = 0
		while n%p==0:
			count += 1
			n //= p
		if count > 0:
			d[p] = count
		if n==1:
			break

	return d


def get_prime_amount_timetable(n, ps):
	timetable = np.zeros((len(ps), )).astype(np.int64)

	for i, p in enumerate(ps, 0):
		t = n//p
		j = 0
		while t > 0:
			t //= p
			j += 1
		timetable[i] = j

	return timetable


def sequence_1():
	n_max = 100
	lst = []

	for n in range(1, n_max+1):
		ps = list(get_primes(n))
		timetable = get_prime_amount_timetable(n, ps).astype(object)
		ps_pow = ps**timetable
		biggest_n_number = np.prod(ps_pow)
		lst.append(biggest_n_number)

	# sequence A003418
	print("lst: {}".format(lst))


def sequence_2():
	n_max = 100
	lst = []

	ps = list(get_primes(n_max))
	for n in range(1, n_max+1):
		prime_factors = np.array(prime_factorization(n, ps))
		print("n: {}".format(n))
		n_mult = np.sum(np.multiply.reduce(prime_factors, axis=1))
		print("n_mult: {}".format(n_mult))
		lst.append(n_mult)

	# sequence A001414
	print("lst: {}".format(lst))


def sequence_3():
	pass


if __name__ == "__main__":
	n_max = 100000
	l_prime = list(get_primes(n_max))
	
	print("n_max: {}".format(n_max))
	print("len(l_prime): {}".format(len(l_prime)))

	sys.exit()

	# sequence_1()
	sequence_2()
