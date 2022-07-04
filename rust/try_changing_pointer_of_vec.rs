mod utils {
	include!("utils.rs");
}

use utils::{ptr_mut_at, val_ref_mut_at};

pub fn print_vec<T: std::fmt::Display>(vec: &Vec<T>) {
	print!("(len: {}, vec: [", vec.len());
	for v in vec {
		print!("{}, ", v);
	}
	print!("])");
}

fn main() {
	type T = u32;

	let mut v_a: Vec<T> = Vec::<T>::new();
	v_a.extend_from_slice(&[1, 2, 3]);

	print!("v_a: ");
	print_vec(&v_a);
	print!("\n");

	let ptr: *mut T = &mut v_a[0];

	let ptr1: *mut T = ptr_mut_at(ptr, 0);
	let ptr2: *mut T = ptr_mut_at(ptr, 1);

	print!("ptr1[1]: {}\n", val_ref_mut_at(ptr1, 1));
	print!("ptr2[0]: {}\n", val_ref_mut_at(ptr2, 0));
	print!("ptr2[-1]: {}\n", val_ref_mut_at(ptr2, -1));

	// 
	let v1: &mut T;
	v1 = val_ref_mut_at(ptr1, 1);
	*v1 = 5;

	unsafe { *(ptr.offset(2)) = 7; }

	print!("ptr1[1]: {}\n", val_ref_mut_at(ptr1, 1));
	print!("ptr2[0]: {}\n", val_ref_mut_at(ptr2, 0));
	print!("ptr2[-1]: {}\n", val_ref_mut_at(ptr2, -1));

	print!("v_a: ");
	print_vec(&v_a);
	print!("\n");
}
