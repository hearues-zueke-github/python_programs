use std::env;
use std::collections::BTreeSet;
use std::collections::BTreeMap;

struct DArr {
	d: i64,
	arr: Vec<i64>,
}

struct DArrSequence {
	d_arr: DArr,
	sequence: Vec<Vec<i64>>,
}

fn calc_cycles(n: i64, modulo: i64) {
	#[inline(always)]
	fn mult(v_1: &Vec<i64>, v_2: &Vec<i64>, v_res: &mut Vec<i64>) {
		assert!(v_1.len() == v_2.len());
		assert!(v_1.len() == v_res.len());

		for i in 0..v_1.len() {
			v_res[i] = v_1[i] * v_2[i];
		}
	}

	#[inline(always)]
	fn sum(v: &Vec<i64>, val_res: &mut i64) {
		*val_res = 0;
		for i in 0..v.len() {
			*val_res += v[i];
		}
	}

	#[inline(always)]
	fn num_to_vec(n: i64, modulo: i64, mut val: i64, v: &mut Vec<i64>) {
		for i in (0..n as usize).rev() {
			v[i] = val % modulo;
			val /= modulo;
		}
	}

	// let d: i64 = 1;
	assert!(n == 2);
	let mut arr: Vec<i64> = vec![0, 0];

	let mut s_tpl_all_orig = BTreeSet::<Vec<i64>>::new();
	for i1 in 0..modulo {
		arr[0] = i1 as i64;
		for i2 in 0..modulo {
			arr[1] = i2 as i64;
			s_tpl_all_orig.insert(vec![i1, i2]);	
		}
	}

	// TODO: make the iteration of the values more generic
	let mut s_t_cycle_all = BTreeSet::<Vec<Vec<i64>>>::new();
	let mod_pow_n = modulo.pow(n as u32);

	for d in 0..mod_pow_n {		
		for i1 in 0..mod_pow_n {
			arr[0] = i1 as i64;
			for i2 in 0..mod_pow_n {
				arr[1] = i2 as i64;

				let mut s_tpl_all = s_tpl_all_orig.clone();
				let mut s_v_v_seq =  BTreeSet::<Vec<Vec<i64>>>::new();

				while s_tpl_all.len() > 0 {
					let mut x = s_tpl_all.iter().next().unwrap().clone();
					s_tpl_all.remove(&x);

					let mut v_seq = Vec::<Vec::<i64>>::new();
					let mut s_tpl = BTreeSet::<Vec::<i64>>::new();

					v_seq.push(x.clone());
					s_tpl.insert(x.clone());

					let mut v_mult = Vec::<i64>::new();
					v_mult.resize(2, 0);
					let mut val_sum: i64 = 0;

					// println!("v_seq: {v_seq:?}");

					let mut is_full_cycle = false;
					loop {
						mult(&x, &arr, &mut v_mult);
						sum(&v_mult, &mut val_sum);
						val_sum += d;
						num_to_vec(n, modulo, val_sum, &mut x);

						if s_tpl.contains(&x) {
							is_full_cycle = true;
							break;
						}
						
						if !s_tpl_all.contains(&x) {
							break;
						}

						v_seq.push(x.clone());
						s_tpl.insert(x.clone());
						s_tpl_all.remove(&x);
					}

					if is_full_cycle {
						// get the idx, where the cycle is beginning
						let idx_cycle = v_seq
	        		.iter()
	        		.position(|v| v == &x)
	        		.unwrap();
						v_seq = v_seq[idx_cycle..].to_vec();

						// find the minimum value in the cycle and shift it by the min_idx,
						// so that the cycle is unique
						let mut min_idx: usize = 0 as usize;
						let mut min_val: &Vec<i64> = &v_seq[0];
						for idx in 1..v_seq.len() {
							let val = &v_seq[idx];
							if min_val > &val {
								min_idx = idx;
								min_val = &val;
							}
						}

						if min_idx > 0 {
							v_seq = [&v_seq[min_idx..], &v_seq[..min_idx]].concat();
						}

						// last insert it into the set of cycles
						s_v_v_seq.insert(v_seq.clone());
					}
				}
				
				// println!("d: {d}, arr: {arr:?}, s_v_v_seq: {s_v_v_seq:?}");

				for v_v_seq in s_v_v_seq {
					if !s_t_cycle_all.contains(&v_v_seq) {
						s_t_cycle_all.insert(v_v_seq);
					}
				}
			}
		}
	}

	// println!("s_t_cycle_all: {s_t_cycle_all:?}");

	let mut d_count_len_cycle = BTreeMap::<usize, i64>::new();
	for t_cycle in &s_t_cycle_all {
		let length = t_cycle.len();
		*d_count_len_cycle.entry(length).or_default() += 1;
	}

	let mut v_count_len_cycle = Vec::<Vec<i64>>::new();
	for (key, value) in &d_count_len_cycle {
		v_count_len_cycle.push(vec![*key as i64, *value]);
	}
	v_count_len_cycle.sort();

	println!("n. {n}, modulo: {modulo}, v_count_len_cycle: {v_count_len_cycle:?}");
}

fn main() {
	let n: i64 = 2;

	let args: Vec<String> = env::args().collect();

	let max_modulo = args[1].to_string().parse::<usize>().unwrap();

	for modulo in 1..=max_modulo {
		calc_cycles(2, modulo as i64);
	}
}
