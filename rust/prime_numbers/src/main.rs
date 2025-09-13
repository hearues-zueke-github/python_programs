use std::env;
use std::fmt;

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;

mod utils;

// check the python implementation from math_numbers/find_prime_mirroring_part.py

fn calc_primes(n: u64) -> Vec<u64> {
	if n < 2 {
		return vec![0u64; 0];
	} else if n < 3 {
		return vec![2u64];
	} else if n < 5 {
		return vec![2u64, 3];
	}

	let mut arr_primes = vec![2u64, 3, 5];
	let d = vec![4u64, 2];

	let mut i: u64 = 7;
	let mut j: usize = 0;

	while i <= n {
		// let q = f64::sqrt(i as f64) as u64 + 1;
		let q = utils::sqrt_u64(i) + 1;

		let mut is_prime = true;

		let mut k: usize = 0;
		let mut v = arr_primes[k];
		while v < q {
			if i % v == 0 {
				is_prime = false;
				break
			}

			k += 1;
			v = arr_primes[k];
		}

		if is_prime {
			arr_primes.push(i)
		}

		i += d[j];
		j = (j + 1) % 2;
	}

	return arr_primes;
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

fn main() {
	let args: Vec<String> = env::args().collect();

	// for number_n in 0..30 {
	// 	let n_sqrt_u64 = utils::sqrt_u64(number_n);
	// 	println!("n: {number_n:}, n_sqrt_u64: {n_sqrt_u64:}");
	// }

	println!("args: {args:?}");

	let n: u64 = args[1].parse::<u64>().unwrap();
	let base_max: u64 = args[2].parse::<u64>().unwrap();
	let file_out_path: &String = &args[3];

	println!("Calc the primes first until {n:}");
	let arr_primes = calc_primes(n);

	let amount_primes = arr_primes.len();
	println!("amount_primes: {amount_primes:}");

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

	// {
	// 	let mut output = File::create(file_out_path).unwrap();
	// 	write!(output, "b1|b2|n1|n1_prim|n2|n2_prim\n").unwrap();
		
	// 	for bpnp in arr_base_pair_number_primes {
	// 		write!(output, "{}|{}|{}|{}|{}|{}\n",
	// 			bpnp.base_pair_number_1.base,
	// 			bpnp.base_pair_number_2.base,
	// 			bpnp.base_pair_number_1.n,
	// 			bpnp.base_pair_number_1.n_prim,
	// 			bpnp.base_pair_number_2.n,
	// 			bpnp.base_pair_number_2.n_prim,
	// 		).unwrap();
	// 	}
	// }
}
