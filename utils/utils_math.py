import sys

import numpy as np

def gen_primes(n):
	if n < 2:
		return
	yield 2
	if n < 3:
		return
	yield 3
	if n < 5:
		return
	yield 5
	l_prime = [2, 3, 5]
	d = [4, 2]

	i = 7
	j = 0
	while i <= n:
		q = int(np.sqrt(i))+1

		is_prime = True

		k = 0
		v = l_prime[k]
		while v < q:
			if i % v == 0:
				is_prime = False
				break
			k += 1
			v = l_prime[k]

		if is_prime:
			yield i
			l_prime.append(i)

		i += d[j]
		j = (j+1) % 2 


def prime_factorization(n, l_prime):
	d = {}

	for p in l_prime:
		count = 0
		while n%p==0:
			count += 1
			n //= p
		if count > 0:
			d[p] = count
		if n==1:
			break

	return d


def get_prime_amount_timetable(n, l_prime):
	timetable = np.zeros((len(l_prime), )).astype(np.int64)

	for i, p in enumerate(l_prime, 0):
		t = n//p
		j = 0
		while t > 0:
			t //= p
			j += 1
		timetable[i] = j

	return timetable
