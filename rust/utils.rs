mod utils {
	#[inline(always)]
	pub fn ptr_mut_at<T>(ptr: *mut T, i: isize) -> *mut T {
		let ret: *mut T;
		unsafe { ret = ptr.offset(i); }
		return ret;
	}

	#[inline(always)]
	pub  fn val_ref_mut_at<T>(ptr: *mut T, i: isize) -> &'static mut T {
		let ret: &mut T;
		unsafe { ret = ptr_mut_at(ptr, i).as_mut().unwrap(); }
		return ret;
	}
}
