include!("../../vec_own.rs");
include!("prng.rs");

use vec_own::VecOwn;
use prng::RandomNumberDevice;

fn main() {
    let arr_seed_u8: VecOwn<u8> = VecOwn::<u8>::new_from_arr(&[0, 1, 2, 3, 4]);
    let length_u8: usize = 128;
    let mut rnd = RandomNumberDevice::new(arr_seed_u8, length_u8);

    rnd.print_current_vals();

    let mut arr_1: VecOwn<u64> = VecOwn::<u64>::new();
    rnd.generate_new_values_u64(&mut arr_1, 1024*1024*4);
    
    let mut arr_2: VecOwn<u64> = VecOwn::<u64>::new();
    rnd.generate_new_values_u64(&mut arr_2, 1024*1024*4);

    let mut arr_3: VecOwn<u64> = VecOwn::<u64>::new();
    rnd.generate_new_values_u64(&mut arr_3, 1024*1024*4);

    let mut arr_4: VecOwn<f64> = VecOwn::<f64>::new();
    rnd.generate_new_values_f64(&mut arr_4, 1024*1024*4);

    // print!("arr_1: {:08X}\n", arr_1);
    // print!("arr_2: {:08X}\n", arr_2);
    // print!("arr_3: {:08X}\n", arr_3);
    // print!("arr_4: {}\n", arr_4);

    print!("arr_1.len(): {}\n", arr_1.len());
    print!("arr_2.len(): {}\n", arr_2.len());
    print!("arr_3.len(): {}\n", arr_3.len());
    print!("arr_4.len(): {}\n", arr_4.len());

    rnd.print_current_vals();
}
