fn f(a: i8) -> i8 {
    return a + 1;
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
}
