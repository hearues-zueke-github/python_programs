struct Currying {
	i i64 [required]
}

fn (self Currying) next_val(v i64) i64 {
	return v + self.i
}

fn main() {
	mut l_fn := []Currying{}

	for i := 0; i < 5; i += 1 {
		l_fn << Currying{i: i}
	}

	num := i64(1234)
	for i, f in l_fn {
		println("i: ${i}, f.next_val(num): ${f.next_val(num)}")
	}
}
