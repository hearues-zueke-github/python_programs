use oorandom;

fn check_vec_sorted(v: &Vec<i64>) -> bool {
    if v.len() < 2 {
        return true;
    }

    for i in 0..v.len()-1 {
        // println!("i: {}", i);
        if v[i] > v[i + 1] {
            return false;
        }
    }

    return true;
}

fn sort_vec_bubble(v: &mut Vec<i64>) {
    if v.len() < 2 {
        return;
    }

    for i in (0..v.len()).rev() {
        for j in 0..i {
            // print!("j: {}", j);
            if v[j] > v[j + 1] {
                // print!(", swap!");
                (v[j], v[j + 1]) = (v[j + 1], v[j])
            }
            // println!("");
        }
    }
}

fn main() {
    let mut a: Vec<i64> = Vec::new();
    let mut rnd = oorandom::Rand64::new(0x1234);

    for _ in 0..10 {
        a.push(rnd.rand_i64());
    }
    println!("a: {:?}", a);

    println!("a.len(): {}", a.len());

    let mut b = a.clone();
    println!("b.len(): {}", b.len());
    println!("b: {:?}", b);
    println!("b is sorted? {}", check_vec_sorted(&b));
    sort_vec_bubble(&mut b);
    println!("b: {:?}", b);
    println!("b is sorted? {}", check_vec_sorted(&b));
}
