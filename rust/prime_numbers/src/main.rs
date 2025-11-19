use std::env;
use std::fmt;
use std::process;

use std::collections::HashSet;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;
use std::path::Path;

mod utils;

// check the python implementation from math_numbers/find_prime_mirroring_part.py

fn calc_primes(n_max: u64) -> Vec<u64> {
	if n_max < 2 {
		return vec![0u64; 0];
	} else if n_max < 3 {
		return vec![2u64];
	} else if n_max < 5 {
		return vec![2u64, 3];
	}

	let mut arr_primes = vec![2u64, 3, 5];
	let arr_jump: Vec<u64> = vec![4u64, 2];
	let mut jump_index: usize = 0;

	let mut next_p: u64 = 7;
	let mut i_q: usize = 1;
	let mut q_pow_2: u64 = arr_primes[i_q] * arr_primes[i_q];

	while next_p <= n_max {
		let mut is_prime: bool = true;

		for i in 2..i_q+1 {
			let v: u64 = arr_primes[i];
			if next_p % v == 0 {
				is_prime = false;
				break
			}
		}

		if is_prime {
			arr_primes.push(next_p)
		}

		next_p += arr_jump[jump_index];

		if next_p >= q_pow_2 {
			i_q += 1;
			let q: u64 = arr_primes[i_q];
			q_pow_2 = q * q;
		}

		jump_index = (jump_index + 1) % 2;
	}

	return arr_primes;
}

fn calc_primes_next(arr_primes: &Vec<u64>, n_min: u64, n_max: u64) -> Vec<u64> {
	let last_prime: u64 = arr_primes[arr_primes.len() - 1];
	if last_prime <= u32::MAX as u64 {
		assert!(last_prime * last_prime >= n_max);
	}

	let mut next_p: u64 = n_min;

	let mut arr_primes_next: Vec<u64> = vec![];
	let l_jump: Vec<u64> = vec![4u64, 2];
	let mut jump_index: usize = 0;
	if next_p % 6 == 0 {
		next_p += 1
	}
	else if next_p % 6 >= 2 {
		next_p += 5 - (next_p % 6);
		jump_index = 1;
	}

	let mut q_pow_2: u64 = 0;

	let mut i_q: usize = 0;
	while i_q < arr_primes.len() {
		let q: u64 = arr_primes[i_q];
		q_pow_2 = q * q;
		
		if q_pow_2 > next_p {
			break;
		}

		i_q += 1;
	}

	while next_p <= n_max {
		let mut is_prime: bool = true;

		for i in 2..i_q+1 {
			let v: u64 = arr_primes[i];
			if next_p % v == 0 {
				is_prime = false;
				break;
			}
		}

		if is_prime {
			arr_primes_next.push(next_p);
		}

		next_p += l_jump[jump_index];

		if next_p >= q_pow_2 {
			i_q += 1;
			let q: u64 = arr_primes[i_q];
			q_pow_2 = q * q;
		}

		jump_index = (jump_index+1) % 2;
	}

	return arr_primes_next
}

struct BasePairNumber {
	base: u64,
	n: u64,
	n_prim: u64,
}

impl fmt::Display for BasePairNumber {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{{b: {}, n: {}, n_prim: {}}}", self.base, self.n, self.n_prim)
	}
}

struct BasePairNumberPrimes {
	base_pair_number_1: BasePairNumber,
	base_pair_number_2: BasePairNumber,
}

impl fmt::Display for BasePairNumberPrimes {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{{b1: {}, b2: {}, n1: {}, n1_prim: {}, n2: {}, n2_prim: {}}}",
			self.base_pair_number_1.base,
			self.base_pair_number_2.base,
			self.base_pair_number_1.n,
			self.base_pair_number_1.n_prim,
			self.base_pair_number_2.n,
			self.base_pair_number_2.n_prim,
		)
	}
}

fn convert_n_to_base_arr_num(mut n: u64, base: u64, arr_num: &mut Vec<u16>) {
	while n > 0 {
		(*arr_num).push((n % base) as u16);
		n /= base;
	}
}

fn convert_base_arr_num_to_n(arr_num: &Vec<u16>, base: u64) -> u64 {
	let mut n_sum: u64 = 0;
	let mut n_pow: u64 = 1;

	for num in arr_num {
		n_sum += (*num as u64) * n_pow;
		n_pow *= base;
	}

	return n_sum;
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_calc_primes() {
		assert_eq!(calc_primes(0), vec![]);
		assert_eq!(calc_primes(1), vec![]);
		assert_eq!(calc_primes(2), vec![2]);
		assert_eq!(calc_primes(3), vec![2, 3]);
		assert_eq!(calc_primes(4), vec![2, 3]);
		assert_eq!(calc_primes(5), vec![2, 3, 5]);
		assert_eq!(calc_primes(10), vec![2, 3, 5, 7]);
		assert_eq!(calc_primes(100), vec![2, 3, 5, 7, 11, 13, 17, 19,
			23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,]);
	}

	#[test]
	fn test_calc_primes_next() {
		let arr_primes = calc_primes(11);
		let arr_primes_next = calc_primes_next(&arr_primes, 12, 100);
		assert_eq!(arr_primes_next, vec![13, 17, 19,
			23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,]);
	}

	#[test]
	fn test_convert_n_to_base_arr_num() {
		let mut arr_num: Vec<u16> = vec![];
		convert_n_to_base_arr_num(86u64, 5, &mut arr_num);
		assert_eq!(arr_num, vec![1u16, 2, 3]);
	}

	#[test]
	fn test_convert_base_arr_num_to_n() {
		let num: u64 = convert_base_arr_num_to_n(&vec![1u16, 2, 3], 5);
		assert_eq!(num, 86u64);
	}
}

fn main() {
	let args: Vec<String> = env::args().collect();

	// for number_n in 0..30 {
	// 	let n_sqrt_u64 = utils::sqrt_u64(number_n);
	// 	println!("n: {number_n:}, n_sqrt_u64: {n_sqrt_u64:}");
	// }

	println!("args: {args:?}");

	let n_max: u64 = args[1].parse::<u64>().unwrap();
	let base_max: u64 = args[2].parse::<u64>().unwrap();
	let file_out_path: &String = &args[3];


	let file_path_arr_primes: &str = "/tmp/arr_primes.bytes";

	if !Path::new(file_path_arr_primes).exists() {
		let n_max_first = 1000000;
		println!("Calculate the first until n_max_first: {n_max_first} primes.");
		let arr_primes_first = calc_primes(n_max_first);
		let mut file_first = OpenOptions::new().write(true).create(true).open(file_path_arr_primes).unwrap();
		let arr_primes_first_vec_u8: &[u8] = unsafe { arr_primes_first.align_to::<u8>().1 };
		println!("Write arr_primes_first into file_first '{file_path_arr_primes}'");
		let _ = file_first.write_all(&arr_primes_first_vec_u8);

		let amount_primes = arr_primes_first.len();
		let last_prime: u64 = *arr_primes_first.last().unwrap();
		println!("amount primes: {amount_primes:}");
		println!("last prime: {last_prime:}");
		process::exit(0);
	}

	let mut file_read = OpenOptions::new().read(true).open(file_path_arr_primes).unwrap();
	let mut arr_primes_vec_u8: Vec<u8> = Vec::new();
	let _ = file_read.read_to_end(&mut arr_primes_vec_u8);
	let mut arr_primes: Vec<u64> = unsafe { (arr_primes_vec_u8[..arr_primes_vec_u8.len()].align_to::<u64>().1).to_vec() };

	let last_calc_prime_prev: u64 = arr_primes[arr_primes.len() - 1];
	let arr_primes_next: Vec<u64> = calc_primes_next(&arr_primes, last_calc_prime_prev+1, last_calc_prime_prev+3_000_000_000u64);
	arr_primes.extend(&arr_primes_next);

	let mut file = OpenOptions::new().write(true).append(true).open(file_path_arr_primes).unwrap();
	let arr_primes_vec_u8: &[u8] = unsafe { arr_primes_next.align_to::<u8>().1 };
	println!("Write arr_primes_first into file '{file_path_arr_primes}'");
	let _ = file.write_all(&arr_primes_vec_u8);

	let amount_primes = arr_primes.len();
	println!("amount_primes: {amount_primes:}");
	let last_calc_prime: u64 = *arr_primes.last().unwrap();
	println!("last_calc_prime: {last_calc_prime:}");

	process::exit(0);

	let mut output = File::create(file_out_path).unwrap();
	write!(output, "b1|b2|n1|n1_prim|n2|n2_prim\n").unwrap();

	// let mut arr_base_pair_number_primes: Vec<BasePairNumberPrimes> = vec![];

	for base_1 in 2..(base_max + 1) {
		let mut hash_set_n_1 = HashSet::<u64>::new();
		let mut arr_base_pair_number: Vec<BasePairNumber> = vec![];
		for n_1 in 1..(amount_primes + 1) {
			if hash_set_n_1.contains(&(n_1 as u64)) {
				continue;
			}

			let mut arr_num_1: Vec<u16> = vec![];
			convert_n_to_base_arr_num(n_1 as u64, base_1, &mut arr_num_1);

			let arr_num_1_prim: Vec<u16> = arr_num_1.iter().copied().rev().collect();
			let n_1_prim = convert_base_arr_num_to_n(&arr_num_1_prim, base_1);

			if arr_num_1 == arr_num_1_prim {
				continue;
			}

			if n_1_prim as usize > amount_primes {
				continue;
			}

			hash_set_n_1.insert(n_1 as u64);
			hash_set_n_1.insert(n_1_prim);

			arr_base_pair_number.push(BasePairNumber {
				base: base_1,
				n: n_1 as u64,
				n_prim: n_1_prim,
			});
		}

		for base_2 in 2..(base_max + 1) {
			for base_pair_number in &arr_base_pair_number {
				let base_1 = base_pair_number.base;
				let n_1 = base_pair_number.n;
				let n_1_prim = base_pair_number.n_prim;

				let n_2: u64 = arr_primes[(n_1 - 1) as usize];
				let n_2_prim: u64 = arr_primes[(n_1_prim - 1) as usize];

				let mut arr_num_2: Vec<u16> = vec![];
				let mut arr_num_2_prim: Vec<u16> = vec![];
				
				convert_n_to_base_arr_num(n_2 as u64, base_2, &mut arr_num_2);
				convert_n_to_base_arr_num(n_2_prim as u64, base_2, &mut arr_num_2_prim);

				if arr_num_2 != arr_num_2_prim.into_iter().rev().collect::<Vec<u16>>() {
					continue;
				}

				let base_pair_number_primes = BasePairNumberPrimes {
					base_pair_number_1: BasePairNumber {base: base_1, n: n_1, n_prim: n_1_prim},
					base_pair_number_2: BasePairNumber {base: base_2, n: n_2, n_prim: n_2_prim},
				};

				println!("{base_pair_number_primes:}");

				// arr_base_pair_number_primes.push(base_pair_number_primes);
				write!(output, "{}|{}|{}|{}|{}|{}\n",
					base_1, base_2,
					n_1, n_1_prim,
					n_2, n_2_prim,
				).unwrap();
			}
		}
	}
}
