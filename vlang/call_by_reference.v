fn foo(mut a []u64) {
	a[0] += 1
}

fn main() {
	mut b := []u64{}
	print("b.len: ${b.len}")

	mut a := [u64(0)]
	println("a: ${a}") foo(mut a) println("a: ${a}")
}
