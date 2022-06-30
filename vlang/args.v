import os

fn main() {
	for i, arg in os.args {
		println("i: ${i}, arg: ${arg}")
	}
}
