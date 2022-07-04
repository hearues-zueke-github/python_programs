use sha2::{Sha256, Digest};

pub fn print_vec<T: std::fmt::Display>(vec: &Vec<T>) {
    print!("(len: {}, vec: [", vec.len());
    for v in vec {
        print!("{}, ", v);
    }
    print!("])");
}

fn main() {
    // println!("Hello World!");
    let mut hasher: Sha256 = Sha256::new();
    let a: &[u8] = &[0, 1, 2];
    hasher.update(a);
    // hasher.update(b"hello world");
    let _result = hasher.finalize();

    println!("_result[0]: {}", _result[0]);
}
