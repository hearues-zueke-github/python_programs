use sha2::{Sha256, Digest};

pub fn print_gen_arr<T: std::fmt::Display + std::fmt::UpperHex>(vec: &[T]) {
    print!("(len: {}, vec: [", vec.len());
    for v in vec {
        print!("{:02X}, ", v);
    }
    print!("])");
}

fn main() {
    let mut hasher_1: Sha256 = Sha256::new();
    let a: &[u8] = &[0, 1, 2, 3];
    hasher_1.update(a);
    let result = hasher_1.finalize();

    print!("result[0]: ");
    print_gen_arr(&result);
    print!("\n");
}
