#pragma once

#include <algorithm>
#include <math.h>
#include <numeric>
#include <vector>

namespace OwnPRNG {
	#include "sha256.h"

	using std::fill;
	using std::iota;
	using std::vector;
	using std::move;

	using fmt::format;
	using fmt::print;

	const size_t block_size = 32;
	const uint64_t mask_uint64_float64 = 0x1fffffffffffffull;
	const double min_val_double = pow(2., -53);

	class RandomNumberDevice {
	public:
		size_t amount_;
		size_t amount_block_;
		size_t amount_vals_;
		vector<uint8_t> vec_state_;
		uint8_t* ptr_state_;
		uint64_t* ptr_state_uint64_t_;

		vector<uint64_t> vec_mult_x_;
		vector<uint64_t> vec_mult_a_;
		vector<uint64_t> vec_mult_b_;

		vector<uint64_t> vec_xor_x_;
		vector<uint64_t> vec_xor_a_;
		vector<uint64_t> vec_xor_b_;

		size_t idx_mult_;
		size_t idx_xor_;

		uint8_t vector_constant_[block_size];
		
		RandomNumberDevice(const size_t amount, const vector<uint8_t> vec_seed);

		void hash_once_loop();
		inline void hash_2_block(const size_t idx_0, const size_t idx_1);
		inline bool are_block_equal(const uint8_t* block_0, const uint8_t* block_1);
		void print_state();
		void print_values();
		void generate_new_values_uint64_t(vector<uint64_t>& vec, const size_t amount);
		void generate_new_values_double(vector<double>& vec, const size_t amount);
	};

	RandomNumberDevice::RandomNumberDevice(const size_t amount, const vector<uint8_t> vec_seed) {
		assert((amount % block_size) == 0);
		assert(amount > block_size);

		amount_ = amount;
		amount_block_ = amount / block_size;
		amount_vals_ = amount / sizeof(uint64_t);
		vec_state_.resize(amount);
		ptr_state_ = &vec_state_[0];
		ptr_state_uint64_t_ = (uint64_t*)ptr_state_;
		fill(ptr_state_, ptr_state_ + amount_, 0);

		const size_t seed_len = vec_seed.size();
		for (size_t i = 0; i < seed_len; ++i) {
			ptr_state_[i % amount_] ^= vec_seed[i];
		}

	  idx_mult_ = 0;
	  idx_xor_ = 0;

		vec_mult_x_.resize(amount_vals_);
		vec_mult_a_.resize(amount_vals_);
		vec_mult_b_.resize(amount_vals_);
		vec_xor_x_.resize(amount_vals_);
		vec_xor_a_.resize(amount_vals_);
		vec_xor_b_.resize(amount_vals_);

		iota(vector_constant_, vector_constant_ + block_size, 1);

		hash_once_loop(); hash_once_loop();
		memcpy(&vec_mult_x_[0], ptr_state_uint64_t_, amount_);
		hash_once_loop(); hash_once_loop();
		memcpy(&vec_mult_a_[0], ptr_state_uint64_t_, amount_);
		hash_once_loop(); hash_once_loop();
		memcpy(&vec_mult_b_[0], ptr_state_uint64_t_, amount_);
		hash_once_loop(); hash_once_loop();
		memcpy(&vec_xor_x_[0], ptr_state_uint64_t_, amount_);
		hash_once_loop(); hash_once_loop();
		memcpy(&vec_xor_a_[0], ptr_state_uint64_t_, amount_);
		hash_once_loop(); hash_once_loop();
		memcpy(&vec_xor_b_[0], ptr_state_uint64_t_, amount_);

		// correct the values for the values a and b for mult and xor
		for (size_t i = 0; i < amount_vals_; ++i) {
			vec_mult_a_[i] += 1 - vec_mult_a_[i] % 4;
			vec_mult_b_[i] += 1 - vec_mult_b_[i] % 2;

			vec_xor_a_[i] += - vec_xor_a_[i] % 2;
			vec_xor_b_[i] += 1 - vec_xor_b_[i] % 2;
		}
	}

	void RandomNumberDevice::hash_once_loop() {
		for (size_t i = 0; i < amount_block_; ++i) {
			hash_2_block((i + 0) % amount_block_, (i + 1) % amount_block_);
		}
	}

	inline void RandomNumberDevice::hash_2_block(const size_t idx_0, const size_t idx_1) {
		uint8_t* block_0 = ptr_state_ + block_size * idx_0;
		uint8_t* block_1 = ptr_state_ + block_size * idx_1;

		if (are_block_equal(block_0, block_1)) {
			for (size_t i = 0; i < block_size; ++i) {
				block_1[i] ^= vector_constant_[i];
			}
		}

		SHA256_CTX ctx_0;
		uint8_t hash_0[32];

		SHA256Init(&ctx_0);
		SHA256Update(&ctx_0, block_0, block_size);
		SHA256Final(&ctx_0, hash_0);

		SHA256_CTX ctx_1;
		uint8_t hash_1[32];

		SHA256Init(&ctx_1);
		SHA256Update(&ctx_1, block_1, block_size);
		SHA256Final(&ctx_1, hash_1);

		for (size_t i = 0; i < block_size; ++i) {
			block_1[i] ^= hash_0[i] ^ hash_1[i] ^ block_0[i];
		}
	}

	inline bool RandomNumberDevice::are_block_equal(const uint8_t* block_0, const uint8_t* block_1) {
		for (size_t i = 0; i < block_size; ++i) {
			if (block_0[i] != block_1[i]) {
				return false;
			}
		}
		return true;
	}

	void RandomNumberDevice::print_state() {
		print("vec_state_:\n");
		for (size_t i = 0; i < amount_block_; ++i) {
			const uint8_t* block = ptr_state_ + block_size * i;
			print("- i: {}, ", i);
			string block_str;
			for (size_t j = 0; j < block_size; ++j) {
				block_str += format("{:02X}", block[j]);
			}
			print("block_str: {}\n", block_str);
		}
	}

	void RandomNumberDevice::print_values() {
		class Record {
		public:
			string name_;
			vector<uint64_t>* vec_ptr_;
		};
		vector<Record> arr = {
			{"vec_mult_x", &vec_mult_x_},
			{"vec_mult_a", &vec_mult_a_},
			{"vec_mult_b", &vec_mult_b_},
			{"vec_xor_x", &vec_xor_x_},
			{"vec_xor_a", &vec_xor_a_},
			{"vec_xor_b", &vec_xor_b_},
		};
		const size_t amount = arr.size();
		print("values:\n");
		for (size_t i = 0; i < amount; ++i) {
			const Record& rec = arr[i];

			print("- name: {}, vec: {}\n", rec.name_, *rec.vec_ptr_);
		}
	}

	void RandomNumberDevice::generate_new_values_uint64_t(vector<uint64_t>& vec, const size_t amount) {
		vec.resize(amount);

		for (size_t i = 0; i < amount; ++i) {
			const uint64_t val_mult_new = ((vec_mult_a_[idx_mult_] * vec_mult_x_[idx_mult_]) + vec_mult_b_[idx_mult_]) ^ vec_xor_x_[idx_xor_];
			vec[i] = val_mult_new;
			vec_mult_x_[idx_mult_] = val_mult_new;

			++idx_mult_;
			if (idx_mult_ >= amount_vals_) {
				idx_mult_ = 0;

				vec_xor_x_[idx_xor_] = (vec_xor_a_[idx_xor_] ^ vec_xor_x_[idx_xor_]) + vec_xor_b_[idx_xor_];

				++idx_xor_;
				if (idx_xor_ >= amount_vals_) {
					idx_xor_ = 0;
				}
			}
		}
	}

	void RandomNumberDevice::generate_new_values_double(vector<double>& vec, const size_t amount) {
		vec.resize(amount);

		vector<uint64_t> vec_uint64_t;
		generate_new_values_uint64_t(vec_uint64_t, amount);

		for (size_t i = 0; i < amount; ++i) {
			vec[i] = min_val_double * (vec_uint64_t[i] & mask_uint64_float64);
		}
	}
}