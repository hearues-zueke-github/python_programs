#! /usr/bin/python3.10

import numpy as np

from hashlib import sha256

# ignore the warnings, for the overflow! should occour sometimes, is intended
np.seterr(all="ignore")

class StateMachine():

	__slots__ = [
		'length',
		'arr_mult_x', 'arr_mult_a', 'arr_mult_b',
		'arr_xor_x', 'arr_xor_a', 'arr_xor_b',
		'idx_values_mult_uint64', 'idx_values_xor_uint64',
	]

	def __init__(self, length):
		self.length = length

		self.arr_mult_x = np.empty((length, ), dtype=np.uint64)
		self.arr_mult_a = np.empty((length, ), dtype=np.uint64)
		self.arr_mult_b = np.empty((length, ), dtype=np.uint64)
		
		self.arr_xor_x = np.empty((length, ), dtype=np.uint64)
		self.arr_xor_a = np.empty((length, ), dtype=np.uint64)
		self.arr_xor_b = np.empty((length, ), dtype=np.uint64)

		self.idx_values_mult_uint64 = 0
		self.idx_values_xor_uint64 = 0


class RandomNumberDevice():

	def __init__(self, arr_seed_uint8, length_uint8=128):
		assert isinstance(arr_seed_uint8, np.ndarray)
		assert len(arr_seed_uint8.shape) == 1
		assert arr_seed_uint8.dtype == np.uint8(0).dtype

		self.length_uint8 = length_uint8
		self.block_size = 32
		assert self.length_uint8 % self.block_size == 0
		self.amount_block = self.length_uint8 // self.block_size

		self.vector_constant = np.arange(1, self.block_size + 1, dtype=np.uint8)

		self.mask_uint64_float64 = np.uint64(0x1fffffffffffff)

		self.arr_seed_uint8 = arr_seed_uint8.copy()

		self.length_values_uint8 = self.length_uint8
		self.length_values_uint64 = self.length_values_uint8 // 8

		self.min_val_float64 = np.float64(2)**-53

		self.init_state()


	def init_state(self):
		self.arr_state_uint8 = np.zeros((self.length_uint8, ), dtype=np.uint8)

		self.arr_state_uint64 = self.arr_state_uint8.view(np.uint64)

		length = self.arr_seed_uint8.shape[0]
		i = 0
		while i < length - self.length_uint8:
			self.arr_state_uint8[:] ^= self.arr_seed_uint8[i:i+self.length_uint8]
			i += self.length_uint8

		if i == 0:
			self.arr_state_uint8[:length] ^= self.arr_seed_uint8
		elif i % self.length_uint8 != 0:
			self.arr_state_uint8[:i%self.length_uint8] ^= self.arr_seed_uint8[i:]
		
		self.sm_curr = StateMachine(length=self.length_values_uint64)
		self.sm_prev = StateMachine(length=self.length_values_uint64)

		# do the double hashing per round, because the avalanche effect should be there, even for the smallest change in each round!
		self.next_hashing_state(); self.next_hashing_state()
		self.sm_curr.arr_mult_x[:] = self.arr_state_uint64
		self.next_hashing_state(); self.next_hashing_state()
		self.sm_curr.arr_mult_a[:] = self.arr_state_uint64
		self.next_hashing_state(); self.next_hashing_state()
		self.sm_curr.arr_mult_b[:] = self.arr_state_uint64
		
		self.next_hashing_state(); self.next_hashing_state()
		self.sm_curr.arr_xor_x[:] = self.arr_state_uint64
		self.next_hashing_state(); self.next_hashing_state()
		self.sm_curr.arr_xor_a[:] = self.arr_state_uint64
		self.next_hashing_state(); self.next_hashing_state()
		self.sm_curr.arr_xor_b[:] = self.arr_state_uint64

		self.sm_curr.arr_mult_a[:] = 1 + self.sm_curr.arr_mult_a - (self.sm_curr.arr_mult_a % 4)
		self.sm_curr.arr_mult_b[:] = 1 + self.sm_curr.arr_mult_b - (self.sm_curr.arr_mult_b % 2)

		self.sm_curr.arr_xor_a[:] = 0 + self.sm_curr.arr_xor_a - (self.sm_curr.arr_xor_a % 2)
		self.sm_curr.arr_xor_b[:] = 1 + self.sm_curr.arr_xor_b - (self.sm_curr.arr_xor_b % 2)

		self.save_current_state_machine_to_previous_state_machine()


	def save_current_state_machine_to_previous_state_machine(self):
		self.sm_prev.arr_mult_x[:] = self.sm_curr.arr_mult_x
		self.sm_prev.arr_mult_a[:] = self.sm_curr.arr_mult_a
		self.sm_prev.arr_mult_b[:] = self.sm_curr.arr_mult_b
		self.sm_prev.arr_xor_x[:] = self.sm_curr.arr_xor_x
		self.sm_prev.arr_xor_a[:] = self.sm_curr.arr_xor_a
		self.sm_prev.arr_xor_b[:] = self.sm_curr.arr_xor_b

		self.sm_prev.idx_values_mult_uint64 = self.sm_curr.idx_values_mult_uint64
		self.sm_prev.idx_values_xor_uint64 = self.sm_curr.idx_values_xor_uint64


	def restore_previous_state_machine_to_current_state_machine(self):
		self.sm_curr.arr_mult_x[:] = self.sm_prev.arr_mult_x
		self.sm_curr.arr_mult_a[:] = self.sm_prev.arr_mult_a
		self.sm_curr.arr_mult_b[:] = self.sm_prev.arr_mult_b
		self.sm_curr.arr_xor_x[:] = self.sm_prev.arr_xor_x
		self.sm_curr.arr_xor_a[:] = self.sm_prev.arr_xor_a
		self.sm_curr.arr_xor_b[:] = self.sm_prev.arr_xor_b

		self.sm_curr.idx_values_mult_uint64 = self.sm_prev.idx_values_mult_uint64
		self.sm_curr.idx_values_xor_uint64 = self.sm_prev.idx_values_xor_uint64


	def print_arr_state_uint8(self):
		print(f"arr_state_uint8:")
		for j in range(0, self.amount_block):
			s = ''.join(map(lambda x: f'{x:02X}', self.arr_state_uint8[self.block_size*(j + 0):self.block_size*(j + 1)]))
			print(f"- j: {j:2}, s: {s}")


	def next_hashing_state(self):
		for i in range(0, self.amount_block):
			idx_blk_0 = (i + 0) % self.amount_block
			idx_blk_1 = (i + 1) % self.amount_block

			idx_0_0 = self.block_size * (idx_blk_0 + 0)
			idx_0_1 = self.block_size * (idx_blk_0 + 1)
			idx_1_0 = self.block_size * (idx_blk_1 + 0)
			idx_1_1 = self.block_size * (idx_blk_1 + 1)
			arr_part_0 = self.arr_state_uint8[idx_0_0:idx_0_1]
			arr_part_1 = self.arr_state_uint8[idx_1_0:idx_1_1]

			if np.all(arr_part_0 == arr_part_1):
				arr_part_1 ^= self.vector_constant

			arr_hash_0 = np.array(list(sha256(arr_part_0.data).digest()), dtype=np.uint8)
			arr_hash_1 = np.array(list(sha256(arr_part_1.data).digest()), dtype=np.uint8)
			self.arr_state_uint8[idx_1_0:idx_1_1] ^= arr_hash_0 ^ arr_hash_1 ^ arr_part_0


	def calc_next_uint64(self, amount):
		assert amount > 0
		arr = np.empty((amount, ), dtype=np.uint64)

		x_xor = self.sm_curr.arr_xor_x[self.sm_curr.idx_values_xor_uint64]
		diff_begin = self.length_values_uint64 - self.sm_curr.idx_values_mult_uint64

		if diff_begin > amount:
			i_from = self.sm_curr.idx_values_mult_uint64
			i_to = i_from + amount

			self.sm_curr.arr_mult_x[i_from:i_to] = ((self.sm_curr.arr_mult_a[i_from:i_to] * self.sm_curr.arr_mult_x[i_from:i_to]) + self.sm_curr.arr_mult_b[i_from:i_to]) ^ x_xor
			arr[:] = self.sm_curr.arr_mult_x[i_from:i_to]

			self.sm_curr.idx_values_mult_uint64 += amount

			return arr

		i = 0
		if self.sm_curr.idx_values_mult_uint64 > 0:
			i_from = self.sm_curr.idx_values_mult_uint64
			i_to = i_from + diff_begin

			self.sm_curr.arr_mult_x[i_from:i_to] = ((self.sm_curr.arr_mult_a[i_from:i_to] * self.sm_curr.arr_mult_x[i_from:i_to]) + self.sm_curr.arr_mult_b[i_from:i_to]) ^ x_xor
			arr[:diff_begin] = self.sm_curr.arr_mult_x[i_from:i_to]

			self.sm_curr.idx_values_mult_uint64 = 0
			i += diff_begin

			a_xor = self.sm_curr.arr_xor_a[self.sm_curr.idx_values_xor_uint64]
			b_xor = self.sm_curr.arr_xor_b[self.sm_curr.idx_values_xor_uint64]
			self.sm_curr.arr_xor_x[self.sm_curr.idx_values_xor_uint64] = (a_xor ^ x_xor) + b_xor

			self.sm_curr.idx_values_xor_uint64 += 1
			if self.sm_curr.idx_values_xor_uint64 >= self.length_values_uint64:
				self.sm_curr.idx_values_xor_uint64 = 0

			x_xor = self.sm_curr.arr_xor_x[self.sm_curr.idx_values_xor_uint64]

		if i >= amount:
			return arr

		while i + self.length_values_uint64 < amount:
			self.sm_curr.arr_mult_x[:] = ((self.sm_curr.arr_mult_a * self.sm_curr.arr_mult_x) + self.sm_curr.arr_mult_b) ^ x_xor
			arr[i:i+self.length_values_uint64] = self.sm_curr.arr_mult_x
			i += self.length_values_uint64

			a_xor = self.sm_curr.arr_xor_a[self.sm_curr.idx_values_xor_uint64]
			b_xor = self.sm_curr.arr_xor_b[self.sm_curr.idx_values_xor_uint64]
			self.sm_curr.arr_xor_x[self.sm_curr.idx_values_xor_uint64] = (a_xor ^ x_xor) + b_xor

			self.sm_curr.idx_values_xor_uint64 += 1
			if self.sm_curr.idx_values_xor_uint64 >= self.length_values_uint64:
				self.sm_curr.idx_values_xor_uint64 = 0

			x_xor = self.sm_curr.arr_xor_x[self.sm_curr.idx_values_xor_uint64]

		diff_rest = amount - i
		i_from = 0
		i_to = diff_rest

		self.sm_curr.arr_mult_x[i_from:i_to] = ((self.sm_curr.arr_mult_a[i_from:i_to] * self.sm_curr.arr_mult_x[i_from:i_to]) + self.sm_curr.arr_mult_b[i_from:i_to]) ^ x_xor
		arr[i:] = self.sm_curr.arr_mult_x[i_from:i_to]

		self.sm_curr.idx_values_mult_uint64 += diff_rest
		if self.sm_curr.idx_values_mult_uint64 >= self.length_values_uint64:
			self.sm_curr.idx_values_mult_uint64 = 0

			a_xor = self.sm_curr.arr_xor_a[self.sm_curr.idx_values_xor_uint64]
			b_xor = self.sm_curr.arr_xor_b[self.sm_curr.idx_values_xor_uint64]
			self.sm_curr.arr_xor_x[self.sm_curr.idx_values_xor_uint64] = (a_xor ^ x_xor) + b_xor

			self.sm_curr.idx_values_xor_uint64 += 1
			if self.sm_curr.idx_values_xor_uint64 >= self.length_values_uint64:
				self.sm_curr.idx_values_xor_uint64 = 0

			x_xor = self.sm_curr.arr_xor_x[self.sm_curr.idx_values_xor_uint64]

		return arr


	def calc_next_uint64_old_slower(self, amount):
		arr = np.empty((amount, ), dtype=np.uint64)

		x_xor = self.sm_curr.arr_xor_x[self.sm_curr.idx_values_xor_uint64]
		for i in range(0, amount):
			x_mult = self.sm_curr.arr_mult_x[self.sm_curr.idx_values_mult_uint64]
			a_mult = self.sm_curr.arr_mult_a[self.sm_curr.idx_values_mult_uint64]
			b_mult = self.sm_curr.arr_mult_b[self.sm_curr.idx_values_mult_uint64]
			x_mult_new = ((a_mult * x_mult) + b_mult) ^ x_xor
			self.sm_curr.arr_mult_x[self.sm_curr.idx_values_mult_uint64] = x_mult_new
			arr[i] = x_mult_new

			self.sm_curr.idx_values_mult_uint64 += 1
			if self.sm_curr.idx_values_mult_uint64 >= self.length_values_uint64:
				self.sm_curr.idx_values_mult_uint64 = 0

				a_xor = self.sm_curr.arr_xor_a[self.sm_curr.idx_values_xor_uint64]
				b_xor = self.sm_curr.arr_xor_b[self.sm_curr.idx_values_xor_uint64]
				self.sm_curr.arr_xor_x[self.sm_curr.idx_values_xor_uint64] = (a_xor ^ x_xor) + b_xor

				self.sm_curr.idx_values_xor_uint64 += 1
				if self.sm_curr.idx_values_xor_uint64 >= self.length_values_uint64:
					self.sm_curr.idx_values_xor_uint64 = 0

				x_xor = self.sm_curr.arr_xor_x[self.sm_curr.idx_values_xor_uint64]

		return arr


	def calc_next_float64(self, amount):
		arr = self.calc_next_uint64(amount=amount)

		return self.min_val_float64 * (arr & self.mask_uint64_float64).astype(np.float64)


if __name__ == '__main__':
	length_uint8 = 128
	rnd = RandomNumberDevice(arr_seed_uint8=np.array([0x01], dtype=np.uint8), length_uint8=length_uint8)

	print(f"arr_1_before before calling calc_next_uint64, rnd.sm_curr.arr_mult_x: {rnd.sm_curr.arr_mult_x}")
	arr_1_before = rnd.calc_next_uint64(amount=1000)
	print(f"arr_1_before after calling calc_next_uint64, rnd.sm_curr.arr_mult_x: {rnd.sm_curr.arr_mult_x}")
	arr_2_before = rnd.calc_next_uint64(amount=1000)
	arr_3_before = rnd.calc_next_uint64(amount=3000)

	rnd.restore_previous_state_machine_to_current_state_machine()

	print(f"arr_1_after before calling calc_next_uint64, rnd.sm_curr.arr_mult_x: {rnd.sm_curr.arr_mult_x}")
	arr_1_after = rnd.calc_next_uint64(amount=1000)
	print(f"arr_1_after after calling calc_next_uint64, rnd.sm_curr.arr_mult_x: {rnd.sm_curr.arr_mult_x}")
	arr_2_after = rnd.calc_next_uint64(amount=1000)
	arr_3_after = rnd.calc_next_uint64(amount=3000)

	assert np.all(arr_1_before==arr_1_after)
	assert np.all(arr_2_before==arr_2_after)
	assert np.all(arr_3_before==arr_3_after)
