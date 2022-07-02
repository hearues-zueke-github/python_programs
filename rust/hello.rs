fn f(a: i8) -> i8 {
	return a + 1;
}

pub struct Row {
	a: i8,
	b: String,
}

impl Row {
	fn print_a(&self) -> () {
		println!("a: {}, yes", self.a);
	}
}

impl std::fmt::Display for Row {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "(value a: {}, value b: {})", self.a, self.b)
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

	let d = Row {
		a: 48,
		b: String::from("Test"),
	};

	println!("d: {}", d);

	d.print_a();
	d.print_a();
}
