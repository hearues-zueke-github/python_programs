fn main() {
	f1 := fn (v i64) i64 {
		return v + 0
	}

	f2 := fn (v i64) i64 {
		return v + 1
	}

	f3 := fn (v i64) i64 {
		return v + 2
	}

	f4 := fn (v i64) i64 {
		return v + 3
	}

	l_fn := [f1, f2, f3, f4]

	num := i64(1234)
	for i, f in l_fn {
		println("i: ${i}, f(num): ${f(num)}")
	}
}
