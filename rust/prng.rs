use std::fmt;
use std::ops;

fn f(a: i8) -> i8 {
	return a + 1;
}

struct VecOwn<T>(Vec<T>);

impl<T> VecOwn<T> {
	fn new() -> Self {
	let vec: Vec<T> = Vec::new();
		return VecOwn::<T>(vec);
	}
}

impl<T> Default for VecOwn<T> {
	fn default() -> Self {
		return Self::new();
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
		// write!(f, "(vals: [0]: {})", self[0])?;
		write!(f, "[")?;
		self.iter().fold(Ok(()), |result, val| {
			result.and_then(|_| write!(f, "{}, ", val))?;
			result
		})?;
		write!(f, "]")?;
		Ok(())
	}
}

#[derive(Default)]
pub struct PRNG {
	v_state: VecOwn<u8>,
	v_a_mult: VecOwn<u64>,
	v_b_mult: VecOwn<u64>,
	v_x_mult: VecOwn<u64>,
	v_a_xor: VecOwn<u64>,
	v_b_xor: VecOwn<u64>,
	v_x_xor: VecOwn<u64>,
}

impl PRNG {
	pub fn new() -> Self {
		let v_state = VecOwn::<u8>::new();
		let v_a_mult = VecOwn::<u64>::new();
		let v_b_mult = VecOwn::<u64>::new();
		let v_x_mult = VecOwn::<u64>::new();
		let v_a_xor = VecOwn::<u64>::new();
		let v_b_xor = VecOwn::<u64>::new();
		let v_x_xor = VecOwn::<u64>::new();

		return Self { v_state, v_a_mult, v_b_mult, v_x_mult, v_a_xor, v_b_xor, v_x_xor };
	}
}

impl fmt::Display for PRNG {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		// write!(f, "")
		// write!(f, "(value v_state: {})", self.v_state)
		write!(f, "(value v_state: {}, value v_a_mult: {})", self.v_state, self.v_a_mult)?;
		Ok(())
	}
}

fn main() {
	let mut a = [1, 2, 3];
	let b = 4;
	println!("Hello World! b: {}", b);	
	for (i, v) in a.iter_mut().enumerate() {
		println!("i: {}, v: {}", i, v);
	}
	let s = format!{"{:?}", a};
	println!("s: {}", s);

	let c = 6;
	let d = f(c);
	println!("d: {}", d);

	let mut d = PRNG::new();

	// println!("before: d: {}", d);

	// TODO: implement the init and create new array functions!

	d.v_state.clear();
	d.v_state.resize(10, 0);

	d.v_state[0] = 10;
	d.v_state[1] = 15;
	d.v_state[9] = 124;


	d.v_a_mult.clear();
	d.v_a_mult.resize(10, 0);

	d.v_a_mult[0] = 143;
	d.v_a_mult[1] = 943;

	d.v_b_mult.clear();
	d.v_b_mult.resize(10, 0);

	d.v_x_mult.clear();
	d.v_x_mult.resize(10, 0);

	d.v_a_xor.clear();
	d.v_a_xor.resize(10, 0);

	d.v_b_xor.clear();
	d.v_b_xor.resize(10, 0);

	d.v_x_xor.clear();
	d.v_x_xor.resize(10, 0);

	// println!("d.v_state: {}", d.v_state);

	println!("after: d: {}", d);

	// d.print_v_state();
	// d.print_v_state();
}
