#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.13 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import decimal
import dill
import gzip
import itertools
import os
import pdb
import re
import sqlite3
import sys
import time
import traceback

import numpy as np # need installation from pip
import pandas as pd # need installation from pip
import multiprocessing as mp

import matplotlib.pyplot as plt # need installation from pip

from collections import defaultdict
from copy import deepcopy, copy
from dataclasses import dataclass
from dotmap import DotMap # need installation from pip
from functools import reduce
from hashlib import sha256
from io import BytesIO
from memory_tempfile import MemoryTempfile # need installation from pip
# from recordclass import RecordClass # need installation from pip
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from tqdm import tqdm # need installation from pip
from PIL import Image

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='prime_numbers_fun', path=os.path.join(PYTHON_PROGRAMS_DIR, "math_numbers/prime_numbers_fun.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

get_primes = prime_numbers_fun.get_primes
get_primes_list = prime_numbers_fun.get_primes_list
get_primes_list_part = prime_numbers_fun.get_primes_list_part


OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

def convert_n_to_base_l_num(n, base):
	l_num = []

	while n > 0:
		l_num.append(n % base)
		n //= base

	return l_num


def convert_base_l_num_to_n(l_num, base):
	n_sum = 0
	n_pow = 1
	for num in l_num:
		n_sum += num * n_pow
		n_pow *= base

	return n_sum

@dataclass(frozen=True)
class BasePairsNumbers:
	b1: int
	b2: int
	n1: int
	n1_prim: int
	n2: int
	n2_prim: int


def do_table_name_exists(con, table_name):
	cur = con.cursor()
	cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
	return table_name in set([v[0] for v in cur])


def main():
	d_arg_name_to_arg_val = {
		'base_max': 10,
	}

	for arg in sys.argv[1:]:
		arg_name, arg_val_str = arg.split('=')
		
		if arg_name == 'base_max':
			d_arg_name_to_arg_val[arg_name] = int(arg_val_str)

	base_max = d_arg_name_to_arg_val['base_max']

	con = sqlite3.connect("prime_numbers.sqlite3")
	cur = con.cursor()

	if not do_table_name_exists(con=con, table_name='table_prime_number'):
		con.execute("""
CREATE TABLE IF NOT EXISTS "table_prime_number" (
	"n"	INTEGER NOT NULL UNIQUE,
	"prime"	INTEGER NOT NULL UNIQUE	
);
""")

	# calc next e.g. 100 primes
		l_prime = get_primes_list(n_max=10000)

		for i, v in enumerate(l_prime, 1):
			cur.execute(f"""
INSERT INTO "table_prime_number" ("n", "prime") VALUES
	({i}, {v});
""")
		con.commit()

	print('Getting already calculated primes.')
	res = cur.execute("SELECT * FROM table_prime_number")
	l_out_1 = list(res)

	print(f'Amount of primes already calculated: {len(l_out_1)}')

	res2 = cur.execute("SELECT MAX(n), prime FROM table_prime_number")
	l_out_2 = list(res2)

	l_prime_prev = [prime for _, prime in l_out_1]

	last_index = l_out_2[0][0]
	prime_max = l_out_2[0][1]

	print(f'last_index: {last_index}, prime_max: {prime_max}')
	print('Calculate new prime numbers')
	l_prime_next = get_primes_list_part(l_prime=l_prime_prev, n_min=prime_max+1, n_max=prime_max+50000000)

	print(f'Last prime calculated: {l_prime_next[-1]}')
	print(f'Inserting amount of new primes: {len(l_prime_next)}')

	print('Insert new calculated prime numbers')
	for i, v in enumerate(l_prime_next, last_index + 1):
		cur.execute(f"""
INSERT INTO "table_prime_number" ("n", "prime") VALUES
	({i}, {v});
""")
	con.commit()

	# globals()['loc'] = locals()
	# sys.exit()

	# concat old and new primes
	l_prime = l_prime_prev + l_prime_next
	# l_prime = [p for p in get_primes(n=n_max)]
	len_l_prime = len(l_prime)

	# arr_heatmap_possible_bases_pair = np.zeros((base_max - 1, base_max - 1))

	d_base_to_len_l_n1_n1_prim = {}
	l_base_pair_numbers = []

	# better approach:
	# calc all pairs in base b of number 1 up to len(l_prime)
	# calc all pairs in base b of prime number 2 up to l_prime[-1]
	# calculate intersection of two bases 1 and 2 where
	# the pair numbers in base b1 and the pair prime numbers in base b2 match

	if not do_table_name_exists(con=con, table_name='base_pairs_from_number_n_to_prime_p'):
		con.execute("""
CREATE TABLE "base_pairs_from_number_n_to_prime_p" (
	"base"	INTEGER NOT NULL,
	"n"	INTEGER NOT NULL,
	"n_prim"	INTEGER NOT NULL,
	"prime_n"	INTEGER NOT NULL,
	"prime_n_prim"	INTEGER NOT NULL,
	UNIQUE(base, n, n_prim)
);
""")

	if not do_table_name_exists(con=con, table_name='base_pairs_from_prime_p_to_number_n'):
		con.execute("""
CREATE TABLE "base_pairs_from_prime_p_to_number_n" (
	"base"	INTEGER NOT NULL,
	"prime"	INTEGER NOT NULL,
	"prime_prim"	INTEGER NOT NULL,
	"n_prime"	INTEGER NOT NULL,
	"n_prime_prim"	INTEGER NOT NULL,
	UNIQUE(base, prime, prime_prim)
);
""")

	for base in range(2, base_max+1):
		print(f'Getting already calculated number prim pair base {base}.')
		res = cur.execute("""
SELECT *
FROM base_pairs_from_number_n_to_prime_p
WHERE
	base = {{base}}
;
""".replace('{{base}}', str(base)))
		l_prev = list(res)

		s_n1 = set()
		for _, n, n_prim, _, _ in l_prev:
			s_n1.add(n)
			s_n1.add(n_prim)

		# check if a db table exists for table_prim_pair_base_{b}
		# l_x1_x1_prim = []
		start_n = 1
		if len(l_prev) > 0:
			start_n = l_prev[-1][1] + 1

		l_n1_n1_prim_next = []
		for n1 in range(start_n, len_l_prime + 1):
			if n1 in s_n1:
				continue

			x1 = convert_n_to_base_l_num(n=n1, base=base)
			x1_prim = x1[::-1]

			if x1 == x1_prim:
				continue

			n1_prim = convert_base_l_num_to_n(l_num=x1_prim, base=base)

			if n1 > len_l_prime or n1_prim > len_l_prime:
				continue

			s_n1.add(n1)
			s_n1.add(n1_prim)

			l_n1_n1_prim_next.append((n1, n1_prim))
			# l_x1_x1_prim.append((x1, x1_prim))

		print(f'Insert new calculated number prim pair base {base}, amount new lines: {len(l_n1_n1_prim_next)}')
		for n_1, n_1_prim in l_n1_n1_prim_next:
			# print(f'base: {base}, n_1: {n_1}, n_1_prim: {n_1_prim}')
			cur.execute("""
INSERT INTO "base_pairs_from_number_n_to_prime_p" ("base", "n", "n_prim", "prime_n", "prime_n_prim") VALUES
	({{base}}, {{n}}, {{n_prim}}, {{prime_n}}, {{prime_n_prim}});
""".replace("{{base}}", str(base)).replace("{{n}}", str(n_1)).replace("{{n_prim}}", str(n_1_prim)).replace("{{prime_n}}", str(l_prime[n_1 - 1])).replace("{{prime_n_prim}}", str(l_prime[n_1_prim - 1])))
		con.commit()

	
	d_prime_to_n = {}
	for n_prime, prime in enumerate(l_prime, 1):
		d_prime_to_n[prime] = n_prime

	for base in range(2, base_max+1):
		print(f'Getting already calculated prime prim pair base {base}.')
		res = cur.execute("""
SELECT *
FROM base_pairs_from_prime_p_to_number_n
WHERE
	base = {{base}}
;
""".replace('{{base}}', str(base)))
		l_prev = list(res)

		s_prime = set()
		for _, prime, prime_prim, _, _ in l_prev:
			s_prime.add(prime)
			s_prime.add(prime_prim)

		start_n_prime = 0

		if len(l_prev) > 0:
			last_prime = max([prime for _, prime, _, _, _ in l_prev])
			start_n_prime = l_prime.index(last_prime) + 1

		l_prime_prime_prim = []
		
		for prime in l_prime[start_n_prime:]:
			if prime in s_prime:
				continue

			l_num = convert_n_to_base_l_num(n=prime, base=base)

			l_num_prim = l_num[::-1]
			if l_num == l_num_prim:
				continue

			prime_prim = convert_base_l_num_to_n(l_num=l_num_prim, base=base)

			if prime_prim in s_prime:
				continue

			s_prime.add(prime)
			s_prime.add(prime_prim)

			if prime_prim in d_prime_to_n:
				l_prime_prime_prim.append((prime, prime_prim))

		print(f'Insert new calculated prime prim pair base {base}, amount new lines: {len(l_prime_prime_prim)}')
		for prime, prime_prim in l_prime_prime_prim:
			# print(f'base: {base}, n_1: {n_1}, n_1_prim: {n_1_prim}')
			cur.execute("""
INSERT INTO "base_pairs_from_prime_p_to_number_n" ("base", "prime", "prime_prim", "n_prime", "n_prime_prim") VALUES
	({{base}}, {{prime}}, {{prime_prim}}, {{n_prime}}, {{n_prime_prim}});
""".replace("{{base}}", str(base)).replace("{{prime}}", str(prime)).replace("{{prime_prim}}", str(prime_prim)).replace("{{n_prime}}", str(d_prime_to_n[prime])).replace("{{n_prime_prim}}", str(d_prime_to_n[prime_prim])))
		con.commit()

	"""
SELECT
	PairNumPrime.base as base_num,
	PairPrimeNum.base AS base_prime,
	PairNumPrime.n,
	PairNumPrime.n_prim,
	PairPrimeNum.prime,
	PairPrimeNum.prime_prim
FROM base_pairs_from_number_n_to_prime_p AS PairNumPrime
INNER JOIN base_pairs_from_prime_p_to_number_n AS PairPrimeNum
WHERE
PairNumPrime.base = 2 AND
PairPrimeNum.base = 2 AND
PairNumPrime.prime_n = PairPrimeNum.prime AND
PairNumPrime.prime_n_prim = PairPrimeNum.prime_prim
;
"""

	"""
SELECT
	T1.base,
	T1.count_pair_prime_num,
	T2.count_pair_num_prime
FROM (
	SELECT
		base,
		count(base) AS count_pair_prime_num
	FROM base_pairs_from_prime_p_to_number_n AS PairPrimeNum
	GROUP BY base
) AS T1
INNER JOIN (
	SELECT
		base,
		count(base) AS count_pair_num_prime
	FROM base_pairs_from_number_n_to_prime_p AS PairNumPrime
	GROUP BY base
) AS T2
WHERE
	T1.base = T2.base
;
"""

	"""
SELECT
	T1.base AS base1,
	T2.base AS base2
FROM (
	SELECT
		base
	FROM base_pairs_from_prime_p_to_number_n AS PairPrimeNum
	GROUP BY base
) AS T1
INNER JOIN (
	SELECT
		base
	FROM base_pairs_from_number_n_to_prime_p AS PairNumPrime
	GROUP BY base
) AS T2
"""

	globals()['loc'] = locals()
	sys.exit()

	for b1 in range(2, base_max + 1):
		if not do_table_name_exists(con=con, table_name=f'table_prim_pair_base_{b1}'):
			con.execute("""
CREATE TABLE IF NOT EXISTS "table_prim_pair_base_{{b1}}" (
	"n_1"	INTEGER NOT NULL UNIQUE,
	"n_1_prim"	INTEGER NOT NULL
);
""".replace('{{b1}}', str(b1)))

		print(f'Getting already calculated prim pair base {b1}.')
		res = cur.execute("SELECT * FROM table_prim_pair_base_{{b1}}".replace('{{b1}}', str(b1)))
		l_n1_n1_prim_prev = list(res)
		
		s_n1 = set()
		for n1, n1_prim in l_n1_n1_prim_prev:
			s_n1.add(n1)
			s_n1.add(n1_prim)

		# check if a db table exists for table_prim_pair_base_{b}
		# l_x1_x1_prim = []
		start_n1 = 1
		if len(l_n1_n1_prim_prev) > 0:
			start_n1 = l_n1_n1_prim_prev[-1][0] + 1

		l_n1_n1_prim_next = []
		for n1 in range(start_n1, len_l_prime + 1):
			if n1 in s_n1:
				continue

			x1 = convert_n_to_base_l_num(n=n1, base=b1)
			x1_prim = x1[::-1]

			if x1 == x1_prim:
				continue

			n1_prim = convert_base_l_num_to_n(l_num=x1_prim, base=b1)

			if n1 > len_l_prime or n1_prim > len_l_prime:
				continue

			s_n1.add(n1)
			s_n1.add(n1_prim)

			l_n1_n1_prim_next.append((n1, n1_prim))
			# l_x1_x1_prim.append((x1, x1_prim))

		print(f'Insert new calculated prim pair base {b1}')
		for n_1, n_1_prim in l_n1_n1_prim_next:
			# print(f'b1: {b1}, n_1: {n_1}, n_1_prim: {n_1_prim}')
			cur.execute("""
INSERT INTO "table_prim_pair_base_{{b1}}" ("n_1", "n_1_prim") VALUES
	({{n_1}}, {{n_1_prim}});
""".replace("{{b1}}", str(b1)).replace("{{n_1}}", str(n_1)).replace("{{n_1_prim}}", str(n_1_prim)))
		con.commit()

		l_n1_n1_prim = l_n1_n1_prim_prev + l_n1_n1_prim_next

		d_base_to_len_l_n1_n1_prim[b1] = len(l_n1_n1_prim)

		for b2 in range(2, base_max + 1):
			for n1, n1_prim in l_n1_n1_prim:
				n2 = l_prime[n1 - 1]
				n2_prim = l_prime[n1_prim - 1]
				
				x2 = convert_n_to_base_l_num(n=n2, base=b2)
				x2_prim = convert_n_to_base_l_num(n=n2_prim, base=b2)

				if x2 != x2_prim[::-1]:
					continue

				# print(f"b1: {b1}, b2: {b2}, n1: {n1}, n1_prim: {n1_prim}, n2: {n2}, n2_prim: {n2_prim}")
				print(f"{{b1: {b1}, b2: {b2}, n1: {n1}, n1_prim: {n1_prim}, n2: {n2}, n2_prim: {n2_prim}}}")

				# arr_heatmap_possible_bases_pair[b1 - 2, b2 - 2] += 1.
				l_base_pair_numbers.append(BasePairsNumbers(b1, b2, n1, n1_prim, n2, n2_prim))

	# plt.imshow(arr_heatmap_possible_bases_pair, cmap='hot', interpolation='nearest')
	# plt.show()

	globals()['loc'] = locals()
	sys.exit()


if __name__ == '__main__':
	main()

	file_path = sys.argv[1]

	with open(file_path, 'r') as f:
		content = f.read()

	l_line = content.split('\n')
	l_column = l_line[0].split('|')

	d_data = {column: [] for column in l_column}
	for line in l_line[1:]:
		if '|' not in line:
			continue

		l_split = line.split('|')

		for column, val_str in zip(l_column, l_split):
			d_data[column].append(int(val_str))

	df = pd.DataFrame(data=d_data, columns=l_column, dtype=object)

	arr_b1 = df['b1'].values
	base_min = np.min(arr_b1)
	base_max = np.max(arr_b1)

	amount_base = base_max - base_min + 1
	arr_history = np.zeros((amount_base, amount_base), dtype=np.int32)

	for b1, b2 in df[['b1', 'b2']].values:
		arr_history[b1 - base_min, b2 - base_min] += 1

	plt.imshow(arr_history, cmap='hot', interpolation='nearest')
	plt.show()

	d_t_b1_b2_to_count = {}
	for b1, b2 in df[['b1', 'b2']].values:
		t = (b1, b2)
		if t not in d_t_b1_b2_to_count:
			d_t_b1_b2_to_count[t] = 0

		d_t_b1_b2_to_count[t] += 1

	d_count_to_t_b1_b2 = {}
	for t_b1_b2, count in d_t_b1_b2_to_count.items():
		if count not in d_count_to_t_b1_b2:
			d_count_to_t_b1_b2[count] = []

		d_count_to_t_b1_b2[count].append(t_b1_b2)
