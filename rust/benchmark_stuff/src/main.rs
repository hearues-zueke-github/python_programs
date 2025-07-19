use std::env;

use std::time::{ Instant, Duration };

use std::collections::HashSet;
use std::collections::BTreeSet;

struct SimplePRNG {
	a: u64,
	b: u64,
	x: u64,
}

impl SimplePRNG {
	fn new(a: u64, b: u64, x: u64) -> SimplePRNG {
		assert!(a % 4 == 1);
		assert!(b % 2 == 1);
		SimplePRNG {
			a: a,
			b: b,
			x: x,
		}
	}

	fn next_val(&mut self) -> u64 {
		self.x = self.x * self.a + self.b;
		self.x
	}
}

#[inline(always)]
fn f() {

}

fn test_hash_set(v_insert: &Vec<u64>, v_remove: &Vec<u64>) -> Duration{
	let now = Instant::now();

	{
		let mut s = HashSet::<u64>::new();
		for val in v_insert {
			s.insert(*val);
		}

		for val in v_remove {
			s.remove(val);
		}
	}

	let elapsed = now.elapsed();
	return elapsed;
}

fn test_btree_set(v_insert: &Vec<u64>, v_remove: &Vec<u64>) -> Duration{
	let now = Instant::now();

	{
		let mut s = BTreeSet::<u64>::new();
		for val in v_insert {
			s.insert(*val);
		}

		for val in v_remove {
			s.remove(val);
		}
	}

	let elapsed = now.elapsed();
	return elapsed;
}

fn main() {
	let mut rng = SimplePRNG::new(0x1245, 0x1239, 0x0);
	let mut v_v = Vec::<Vec<u64>>::new();

	let args: Vec<String> = env::args().collect();
	let n = args[1].to_string().parse::<usize>().unwrap();

	for _ in 0..5 {
		let mut v = Vec::<u64>::new();
		
		for _ in 0..n {
			v.push(rng.next_val());
		}

		v_v.push(v);
	}

	println!("n: {n}");

	println!("Test HashSet.");
	let mut v_time_hash_set = Vec::<Duration>::new();
	for v in &v_v {
		let elapsed = test_hash_set(&v, &v);
		v_time_hash_set.push(elapsed);
	}

	println!("Test BTreeSet.");
	let mut v_time_btree_set = Vec::<Duration>::new();
	for v in &v_v {
		let elapsed = test_btree_set(&v, &v);
		v_time_btree_set.push(elapsed);
	}

	let mut time_hash_set_sum = Duration::default();
	for elapsed in &v_time_hash_set {
		time_hash_set_sum += *elapsed;
	}
	let time_hash_set_mean = time_hash_set_sum / v_time_hash_set.len() as u32;

	let mut time_btree_set_sum = Duration::default();
	for elapsed in &v_time_btree_set {
		time_btree_set_sum += *elapsed;
	}
	let time_btree_set_mean = time_btree_set_sum / v_time_btree_set.len() as u32;


	println!("time_hash_set_mean: {time_hash_set_mean:?}");
	println!("time_btree_set_mean: {time_btree_set_mean:?}");
}
