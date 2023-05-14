use oorandom;
use std::time;

// use num_traits;
// use core::num::FpCategory::Zero;

fn check_vec_sorted(v: &Vec<i64>) -> bool {
	if v.len() < 2 {
		return true;
	}

	for i in 0..v.len()-1 {
		if v[i] > v[i + 1] {
			return false;
		}
	}

	return true;
}

fn sort_vec_shellshort_1(v: &mut Vec<i64>) {
	if v.len() < 2 {
		return;
	}

	let mut v_jump: Vec<usize> = vec![1];
	let size = v.len();
	let mut k: u32 = 1;
	loop {
		let jump = usize::pow(4, k) + 3 * usize::pow(2, k-1) + 1;
		if jump >= size.try_into().unwrap() {
			break;
		}
		v_jump.push(jump);
		k += 1;
	}

	for jump in v_jump.iter().rev() {
		for jump_idx in 1..jump+1 {
			for i in (jump_idx..size).step_by(*jump) {
				let mut j = i;
				while j > jump - 1 {
					let j_prev = j - jump;
					if v[j] < v[j_prev] {
						(v[j], v[j_prev]) = (v[j_prev], v[j]);
						j = j_prev;
					} else {
						break;
					}
				}
			}
		}
	}

	for i in 1..size {
		let mut j = i;
		while j > 0 {
			if v[j] < v[j-1] {
				(v[j], v[j - 1]) = (v[j - 1], v[j]);
				j -= 1;
			} else {
				break;
			}
		}
	}
}

fn sort_quick(array: &mut Vec<i64>) {
	let start = 0;
	let end = array.len() - 1;
	quick_sort_partition(array, start, end as isize);
}

fn quick_sort_partition(array: &mut Vec<i64>, start: isize, end: isize) {
	if start < end && end - start >= 1 {
		let pivot = partition(array, start as isize, end as isize);
		quick_sort_partition(array, start, pivot - 1);
		quick_sort_partition(array, pivot + 1, end);
	}
}

fn partition(array: &mut Vec<i64>, l: isize, h: isize) -> isize {
	let pivot = array[h as usize];
	let mut i = l - 1; // Index of the smaller element

	for j in l..h {
		if array[j as usize] <= pivot {
			i = i + 1;
			array.swap(i as usize, j as usize);
		}
	}

	array.swap((i + 1) as usize, h as usize);

	i + 1
}

fn merge(left: &Vec<i64>, right: &Vec<i64>) -> Vec<i64> {
	let mut i = 0;
	let mut j = 0;
	let mut merged: Vec<i64> = Vec::new();

	while i < left.len() && j < right.len() {
		if left[i] < right[j] {
			merged.push(left[i]);
			i = i + 1;
		} else {
			merged.push(right[j]);
			j = j + 1;
		}
	}

	if i < left.len() {
		while i < left.len() {
			merged.push(left[i]);
			i = i + 1;
		}
	}

	if j < right.len() {
		while j < right.len() {
			merged.push(right[j]);
			j = j + 1;
		}
	}

	merged
}

fn sort_merge(vec: &Vec<i64>) -> Vec<i64> {
	if vec.len() < 2 {
		vec.to_vec()
	} else {
		let size = vec.len() / 2;
		let left = sort_merge(&vec[0..size].to_vec());
		let right = sort_merge(&vec[size..].to_vec());
		let merged = merge(&left, &right);

		merged
	}
}

mod my_sorting {
	struct Range {
		idx_1: usize,
		idx_2: usize,
	}

	enum State {
		IsLe,
		IsGt,
		SetNextVal,
	}

	pub fn sort_merge_own(vec: &Vec<i64>) -> Vec<i64> {
		if vec.len() < 2 {
			return vec.clone();
		}

		let mut vec_1: Vec<i64> = vec.clone();
		let mut vec_2: Vec<i64> = vec![0; vec.len()];
		
		return vec.clone();
	}
}

fn mean(v: &Vec<f64>) -> f64 {
	let mut sum: f64 = 0.0;
	for x in v {
		sum += x;
	}
	return sum / v.len() as f64;
}

#[inline(always)]
fn sort_func_1(vec_src: &Vec<i64>, vec_dst: &mut Vec<i64>) {
	*vec_dst = vec_src.clone();
	sort_quick(vec_dst);
}

#[inline(always)]
fn sort_func_2(vec_src: &Vec<i64>, vec_dst: &mut Vec<i64>) {
	*vec_dst = vec_src.clone();
	sort_vec_shellshort_1(vec_dst);
}

#[inline(always)]
fn sort_func_3(vec_src: &Vec<i64>, vec_dst: &mut Vec<i64>)	{
	*vec_dst = vec_src.clone();
	vec_dst.sort();
}

#[inline(always)]
fn sort_func_4(vec_src: &Vec<i64>, vec_dst: &mut Vec<i64>)	{
	*vec_dst = sort_merge(&vec_src);
}

#[inline(always)]
fn sort_func_5(vec_src: &Vec<i64>, vec_dst: &mut Vec<i64>)	{
	*vec_dst = my_sorting::sort_merge_own(&vec_src);
}

struct SortingFunction {
	func: fn(&Vec<i64>, &mut Vec<i64>),
	name: String,
}


mod my {
	use num_traits::{Zero, One};

	pub struct Counter<T> {
		count: T,
	}

	impl<T: std::ops::AddAssign + Copy + One + Zero> Counter<T> {
		#[inline(always)]
		pub fn new() -> Counter<T> {
			return Counter { count: T::zero() };
		}

		#[inline(always)]
		pub fn get_count(&mut self) -> T {
			return self.count;
		}

		#[inline(always)]
		pub fn increment(&mut self) {
			self.count += T::one();
		}
	}
}

fn main() {
	let vec_sorting_func: Vec<SortingFunction> = vec![
		SortingFunction{func: sort_func_1, name: String::from("Quick Sort")},
		SortingFunction{func: sort_func_2, name: String::from("Shell Sort 1")},
		SortingFunction{func: sort_func_3, name: String::from("Rust Sort")},
		SortingFunction{func: sort_func_4, name: String::from("Merge Sort")},
		SortingFunction{func: sort_func_5, name: String::from("Merge Own Sort")},
	];

	let mut vec_vec_time: Vec<Vec<f64>> = vec![];
	
	for _ in &vec_sorting_func {
		vec_vec_time.push(Vec::<f64>::new());
	}

	let amount_loops: i64 = 10;
	let length: i64 = 500000;

	let mut counter = my::Counter::<u128>::new();
	for _ in 0..amount_loops {
		counter.increment();
		let mut rnd = oorandom::Rand64::new(0x51245 + counter.get_count());

		let mut vec_base: Vec<i64> = Vec::new();
		for _ in 0..length {
			vec_base.push(rnd.rand_i64());
		}

		println!();
		let mut vec_temp = vec![];
		for (i, sorting_func) in vec_sorting_func.iter().enumerate() {
			let now = time::Instant::now();
			(sorting_func.func)(&vec_base, &mut vec_temp);
			let elapsed = now.elapsed().subsec_nanos();
			println!("sort name: '{}', elapsed: {} ns", sorting_func.name, elapsed);
			assert!(check_vec_sorted(&vec_temp));
			vec_vec_time[i].push(elapsed as f64 / 1_000_000_000.);
		}
	}

	println!();
	println!("Needed times:");
	for (i, sorting_func) in vec_sorting_func.iter().enumerate() {
		let mean_time = mean(&vec_vec_time[i]);
		println!("sort name: {}, mean_time: {:.6} s", sorting_func.name, mean_time);
	}
}
