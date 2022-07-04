use std::fmt;
use std::ops;

use std::convert::TryInto;
use sha2::{Sha256, Digest};

mod utils {
	include!("utils.rs");
}

use utils::{ptr_mut_at, val_ref_mut_at};

pub struct VecOwn<T>(Vec<T>);

impl<T> VecOwn<T> {
	fn new() -> Self {
		let vec: Vec<T> = Vec::new();
		return VecOwn::<T>(vec);
	}

	fn new_from_vec(vec: Vec<T>) -> Self {
		return VecOwn::<T>(vec);
	}
}

impl<T> std::cmp::PartialEq for VecOwn<T> {
	fn eq(&self, other: &VecOwn<T>) -> bool {
		if self.len() != other.len() {
			return false;
		}

		return true;
	}

    fn ne(&self, other: &VecOwn<T>) -> bool {
    	return false;
    }
}

impl<T: std::clone::Clone> VecOwn<T> {
	fn new_from_arr(arr: &[T]) -> Self {
		let vec: Vec<T> = arr.to_vec();
		return VecOwn::<T>(vec);
	}
}

impl<T> Default for VecOwn<T> {
	fn default() -> Self {
		return Self::new();
	}
}

impl<T: std::clone::Clone> Clone for VecOwn<T> {
	fn clone(&self) -> Self {
		return Self(Vec::<T>::clone(&self.0));
	}
}

impl<T> ops::Deref for VecOwn<T> {
	type Target = Vec<T>;
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl<T> ops::DerefMut for VecOwn<T> {
	 fn deref_mut(&mut self) -> &mut Self::Target {
		 &mut self.0
	 }
}

impl<T: fmt::Display> fmt::Display for VecOwn<T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "[")?;
		for v in self.iter() {
			write!(f, "{}, ", v)?;
		}
		write!(f, "]")?;
		Ok(())
	}
}

#[derive(Default)]
pub struct StateMachine {
	v_a_mult: VecOwn<u64>,
	v_b_mult: VecOwn<u64>,
	v_x_mult: VecOwn<u64>,
	v_a_xor: VecOwn<u64>,
	v_b_xor: VecOwn<u64>,
	v_x_xor: VecOwn<u64>,
	idx_values_mult_u64: u32,
	idx_values_xor_u64: u32,
}

impl StateMachine {
	pub fn new() -> Self {
		let v_a_mult = VecOwn::<u64>::new();
		let v_b_mult = VecOwn::<u64>::new();
		let v_x_mult = VecOwn::<u64>::new();
		let v_a_xor = VecOwn::<u64>::new();
		let v_b_xor = VecOwn::<u64>::new();
		let v_x_xor = VecOwn::<u64>::new();

		let idx_values_mult_u64 = 0;
		let idx_values_xor_u64 = 0;

		return Self {
			v_a_mult,
			v_b_mult,
			v_x_mult,
			v_a_xor,
			v_b_xor,
			v_x_xor,
			idx_values_mult_u64,
			idx_values_xor_u64,
		};
	}

	pub fn init_vals(&mut self, size: usize) {
		self.v_a_mult.resize(size, 0);
		self.v_b_mult.resize(size, 0);
		self.v_x_mult.resize(size, 0);
		self.v_a_xor.resize(size, 0);
		self.v_b_xor.resize(size, 0);
		self.v_x_xor.resize(size, 0);

		self.idx_values_mult_u64 = 0;
		self.idx_values_xor_u64 = 0;
	}
}

#[derive(Default)]
pub struct RandomNumberDevice {
	v_state_u8: VecOwn<u8>,
	block_size: usize,
	length_u8: usize,
	length_u64: usize,
	amount_block: usize,
	mask_u64_f64: u64,
	arr_seed_u8: VecOwn<u8>,
	min_val_f64: f64,
	state_prev: StateMachine,
	state_curr: StateMachine,
}

impl RandomNumberDevice {
	pub fn new(arr_seed_u8_: VecOwn<u8>, length_u8_: usize) -> Self {
		let v_state_u8: VecOwn<u8> = VecOwn::<u8>::new();
		let block_size: usize = 32;
		assert!(length_u8_ % block_size == 0);
		let length_u8: usize = length_u8_;
		let length_u64: usize = length_u8 / 8;
		let amount_block: usize = length_u8_ / block_size;
		let mask_u64_f64: u64 = 0x1fffffffffffff;
		let arr_seed_u8: VecOwn<u8> = arr_seed_u8_.clone();
		let min_val_f64: f64 = 2.0e-53;
		let state_prev: StateMachine = StateMachine::new();
		let state_curr: StateMachine = StateMachine::new();

		let mut self_ = Self {
			v_state_u8,
			block_size,
			length_u8,
			length_u64,
			amount_block,
			mask_u64_f64,
			arr_seed_u8,
			min_val_f64: min_val_f64,
			state_prev,
			state_curr,
		};

		self_.init_state();

		return self_;
	}

	pub fn init_state(&mut self) {
		self.v_state_u8.resize(self.length_u8, 0);
		self.v_state_u8.fill(0);

		let length: usize = self.arr_seed_u8.len();
		let mut i: usize = 0;
		while length - i > self.length_u8 {
			for j in 0..self.length_u8 {
				self.v_state_u8[j] ^= self.arr_seed_u8[i+j];
			}
			i += self.length_u8;
		}

		if i == 0 {
			for j in 0..length {
				self.v_state_u8[j] ^= self.arr_seed_u8[j];
			}
		} else if i % self.length_u8 != 0 {
			for j in 0..(i % self.length_u8) {
				self.v_state_u8[j] ^= self.arr_seed_u8[i + j];
			}
		}

		self.state_curr.init_vals(self.length_u64);
		self.state_prev.init_vals(self.length_u64);

		print!("self.v_state_u8: {}\n", self.v_state_u8);

		self.next_hashing_state(); self.next_hashing_state();
		// self.sm_curr.arr_mult_x[..] = self.v_state_u64;
		self.next_hashing_state(); self.next_hashing_state();
		// self.sm_curr.arr_mult_a[:] = self.v_state_u64;
		self.next_hashing_state(); self.next_hashing_state();
		// self.sm_curr.arr_mult_b[:] = self.v_state_u64;
		self.next_hashing_state(); self.next_hashing_state();

		// self.sm_curr.arr_xor_x[:] = self.v_state_u64;
		self.next_hashing_state(); self.next_hashing_state();
		// self.sm_curr.arr_xor_a[:] = self.v_state_u64;
		self.next_hashing_state(); self.next_hashing_state();
		// self.sm_curr.arr_xor_b[:] = self.v_state_u64;

		// self.sm_curr.arr_mult_a[:] = 1 + self.sm_curr.arr_mult_a - (self.sm_curr.arr_mult_a % 4);
		// self.sm_curr.arr_mult_b[:] = 1 + self.sm_curr.arr_mult_b - (self.sm_curr.arr_mult_b % 2);

		// self.sm_curr.arr_xor_a[:] = 0 + self.sm_curr.arr_xor_a - (self.sm_curr.arr_xor_a % 2);
		// self.sm_curr.arr_xor_b[:] = 1 + self.sm_curr.arr_xor_b - (self.sm_curr.arr_xor_b % 2);

		self.save_current_state_machine_to_previous_state_machine();
	}

	pub fn next_hashing_state(&mut self) {
		let ptr: *mut u8 = &mut self.v_state_u8[0];
		for i in 0..self.amount_block {			
			let idx_blk_0: usize = (i + 0) % self.amount_block;
			let idx_blk_1: usize = (i + 1) % self.amount_block;

			let idx_0_0: usize = self.block_size * (idx_blk_0 + 0);
			let idx_0_1: usize = self.block_size * (idx_blk_0 + 1);
			let idx_1_0: usize = self.block_size * (idx_blk_1 + 0);
			let idx_1_1: usize = self.block_size * (idx_blk_1 + 1);
			// let arr_part_0: &[u8] = &self.v_state_u8[idx_0_0..idx_0_1];
			// let arr_part_1: &mut[u8] = &mut self.v_state_u8[idx_1_0..idx_1_1];

			let ptr_0: *mut u8 = ptr_mut_at(ptr, idx_0_0.try_into().unwrap());
			let ptr_1: *mut u8 = ptr_mut_at(ptr, idx_1_0.try_into().unwrap());

			let mut is_all_equal: bool = true;

			for j in 0..self.amount_block {
				unsafe {
					if *val_ref_mut_at(ptr_0, j.try_into().unwrap()) != *val_ref_mut_at(ptr_1, j.try_into().unwrap()) {
						is_all_equal = false;
						break;
					}
				}
			}

			if is_all_equal {
				let mut v = 0x01u8;
				for j in 0..self.block_size {
					*val_ref_mut_at(ptr_1, j.try_into().unwrap()) ^= v;
					v += 1;
				}
			}

			

			// arr_hash_0 = np.array(list(sha256(arr_part_0.data).digest()), dtype=np.uint8);
			// arr_hash_1 = np.array(list(sha256(arr_part_1.data).digest()), dtype=np.uint8);
			// self.v_state_u8[idx_1_0:idx_1_1] ^= arr_hash_0 ^ arr_hash_1 ^ arr_part_0;
		}
	}

	pub fn save_current_state_machine_to_previous_state_machine(&mut self) {

	}
}

impl fmt::Display for RandomNumberDevice {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "RandomNumberDevice(\n")?;
		write!(f, "  v_state_u8: {},\n", self.v_state_u8)?;
		write!(f, "  length_u8: {},\n", self.length_u8)?;
		write!(f, ")")?;
		Ok(())
	}
}

fn main() {
	let arr_seed_u8: VecOwn<u8> = VecOwn::<u8>::new_from_arr(&[9, 1, 2, 3]);
	let length_u8: usize = 64;
	let mut d = RandomNumberDevice::new(arr_seed_u8, length_u8);

	// println!("before: d: {}", d);

	// TODO: implement the init and create new array functions!

	d.v_state_u8.clear();
	d.v_state_u8.resize(10, 0);

	// d.v_state_u8[0] = 10;
	// d.v_state_u8[1] = 15;
	// d.v_state_u8[9] = 124;


	// d.v_a_mult.clear();
	// d.v_a_mult.resize(10, 0);

	// d.v_a_mult[0] = 143;
	// d.v_a_mult[1] = 943;

	// d.v_b_mult.clear();
	// d.v_b_mult.resize(10, 0);

	// d.v_x_mult.clear();
	// d.v_x_mult.resize(10, 0);

	// d.v_a_xor.clear();
	// d.v_a_xor.resize(10, 0);

	// d.v_b_xor.clear();
	// d.v_b_xor.resize(10, 0);

	// d.v_x_xor.clear();
	// d.v_x_xor.resize(10, 0);

	// println!("d.v_state_u8: {}", d.v_state_u8);

	println!("after: d: {}", d);

	// d.print_v_state();
	// d.print_v_state();
}
