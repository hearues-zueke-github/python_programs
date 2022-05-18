import numpy as np

from hashlib import sha256

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
		self.idx_values_mult_uint64 = 0
		self.idx_values_xor_uint64 = 0
		self.idx_values_xor_uint64 = 0

		self.arr_mult_x = np.zeros((self.length_values_uint64, ), dtype=np.uint64)
		self.arr_mult_a = np.zeros((self.length_values_uint64, ), dtype=np.uint64)
		self.arr_mult_b = np.zeros((self.length_values_uint64, ), dtype=np.uint64)

		self.arr_xor_x = np.zeros((self.length_values_uint64, ), dtype=np.uint64)
		self.arr_xor_a = np.zeros((self.length_values_uint64, ), dtype=np.uint64)
		self.arr_xor_b = np.zeros((self.length_values_uint64, ), dtype=np.uint64)

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
		
		# do the double hashing per round, because the avalanche effect should be there, even for the smallest change in each round!
		self.next_hashing_state(); self.next_hashing_state()
		self.arr_mult_x[:] = self.arr_state_uint64
		self.next_hashing_state(); self.next_hashing_state()
		self.arr_mult_a[:] = self.arr_state_uint64
		self.next_hashing_state(); self.next_hashing_state()
		self.arr_mult_b[:] = self.arr_state_uint64
		
		self.next_hashing_state(); self.next_hashing_state()
		self.arr_xor_x[:] = self.arr_state_uint64
		self.next_hashing_state(); self.next_hashing_state()
		self.arr_xor_a[:] = self.arr_state_uint64
		self.next_hashing_state(); self.next_hashing_state()
		self.arr_xor_b[:] = self.arr_state_uint64

		self.arr_mult_a[:] = 1 + self.arr_mult_a - (self.arr_mult_a % 4)
		self.arr_mult_b[:] = 1 + self.arr_mult_b - (self.arr_mult_b % 2)

		self.arr_xor_a[:] = 0 + self.arr_xor_a - (self.arr_xor_a % 2)
		self.arr_xor_b[:] = 1 + self.arr_xor_b - (self.arr_xor_b % 2)


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

		x_xor = self.arr_xor_x[self.idx_values_xor_uint64]
		diff_begin = self.length_values_uint64 - self.idx_values_mult_uint64

		if diff_begin > amount:
			i_from = self.idx_values_mult_uint64
			i_to = i_from + amount

			self.arr_mult_x[i_from:i_to] = ((self.arr_mult_a[i_from:i_to] * self.arr_mult_x[i_from:i_to]) + self.arr_mult_b[i_from:i_to]) ^ x_xor
			arr[:] = self.arr_mult_x[i_from:i_to]

			self.idx_values_mult_uint64 += amount

			return arr

		i = 0
		if self.idx_values_mult_uint64 > 0:
			i_from = self.idx_values_mult_uint64
			i_to = i_from + diff_begin

			self.arr_mult_x[i_from:i_to] = ((self.arr_mult_a[i_from:i_to] * self.arr_mult_x[i_from:i_to]) + self.arr_mult_b[i_from:i_to]) ^ x_xor
			arr[:diff_begin] = self.arr_mult_x[i_from:i_to]

			self.idx_values_mult_uint64 = 0
			i += diff_begin

			a_xor = self.arr_xor_a[self.idx_values_xor_uint64]
			b_xor = self.arr_xor_b[self.idx_values_xor_uint64]
			self.arr_xor_x[self.idx_values_xor_uint64] = (a_xor ^ x_xor) + b_xor

			self.idx_values_xor_uint64 += 1
			if self.idx_values_xor_uint64 >= self.length_values_uint64:
				self.idx_values_xor_uint64 = 0

			x_xor = self.arr_xor_x[self.idx_values_xor_uint64]

		if i >= amount:
			return arr

		while i + self.length_values_uint64 < amount:
			self.arr_mult_x[:] = ((self.arr_mult_a * self.arr_mult_x) + self.arr_mult_b) ^ x_xor
			arr[i:i+self.length_values_uint64] = self.arr_mult_x
			i += self.length_values_uint64

			a_xor = self.arr_xor_a[self.idx_values_xor_uint64]
			b_xor = self.arr_xor_b[self.idx_values_xor_uint64]
			self.arr_xor_x[self.idx_values_xor_uint64] = (a_xor ^ x_xor) + b_xor

			self.idx_values_xor_uint64 += 1
			if self.idx_values_xor_uint64 >= self.length_values_uint64:
				self.idx_values_xor_uint64 = 0

			x_xor = self.arr_xor_x[self.idx_values_xor_uint64]

		diff_rest = amount - i
		i_from = 0
		i_to = diff_rest

		self.arr_mult_x[i_from:i_to] = ((self.arr_mult_a[i_from:i_to] * self.arr_mult_x[i_from:i_to]) + self.arr_mult_b[i_from:i_to]) ^ x_xor
		arr[i:] = self.arr_mult_x[i_from:i_to]

		self.idx_values_mult_uint64 += diff_rest
		if self.idx_values_mult_uint64 >= self.length_values_uint64:
			self.idx_values_mult_uint64 = 0

			a_xor = self.arr_xor_a[self.idx_values_xor_uint64]
			b_xor = self.arr_xor_b[self.idx_values_xor_uint64]
			self.arr_xor_x[self.idx_values_xor_uint64] = (a_xor ^ x_xor) + b_xor

			self.idx_values_xor_uint64 += 1
			if self.idx_values_xor_uint64 >= self.length_values_uint64:
				self.idx_values_xor_uint64 = 0

			x_xor = self.arr_xor_x[self.idx_values_xor_uint64]

		return arr


	def calc_next_uint64_old_slower(self, amount):
		arr = np.empty((amount, ), dtype=np.uint64)

		x_xor = self.arr_xor_x[self.idx_values_xor_uint64]
		for i in range(0, amount):
			x_mult = self.arr_mult_x[self.idx_values_mult_uint64]
			a_mult = self.arr_mult_a[self.idx_values_mult_uint64]
			b_mult = self.arr_mult_b[self.idx_values_mult_uint64]
			x_mult_new = ((a_mult * x_mult) + b_mult) ^ x_xor
			self.arr_mult_x[self.idx_values_mult_uint64] = x_mult_new
			arr[i] = x_mult_new

			self.idx_values_mult_uint64 += 1
			if self.idx_values_mult_uint64 >= self.length_values_uint64:
				self.idx_values_mult_uint64 = 0

				a_xor = self.arr_xor_a[self.idx_values_xor_uint64]
				b_xor = self.arr_xor_b[self.idx_values_xor_uint64]
				self.arr_xor_x[self.idx_values_xor_uint64] = (a_xor ^ x_xor) + b_xor

				self.idx_values_xor_uint64 += 1
				if self.idx_values_xor_uint64 >= self.length_values_uint64:
					self.idx_values_xor_uint64 = 0

				x_xor = self.arr_xor_x[self.idx_values_xor_uint64]

		return arr


	def calc_next_float64(self, amount):
		arr = self.calc_next_uint64(amount=amount)

		return self.min_val_float64 * (arr & self.mask_uint64_float64).astype(np.float64)
