module main

import hash

fn test_calc_simple_hash() {

}

fn test_main() {
	mut map_key_val := MapKeyArrayValueArray[u8, u16]{}

	map_key_val.put_key_val([u8(3)], [u16(4)])
	map_key_val.put_key_val([u8(4)], [u16(4)])
	map_key_val.put_key_val([u8(7)], [u16(3)])
	map_key_val.put_key_val([u8(5), u8(5), u8(188)], [u16(9)])
	map_key_val.put_key_val([u8(5), u8(5), u8(237)], [u16(7)])
	map_key_val.print()
	
	map_key_val.delete_key([u8(5), u8(5), u8(237)])
	map_key_val.print()

	mut a1 := map_key_val.get_val([u8(5), u8(5), u8(188)])!
	a1[0] = 45
	println('a1: ${a1}')

	arr_1 := [u64(3), 4, 5, 6, 7]
	arr_2 := [u64(3), 4, 5, 6, 7]

	println('arr_1: ${arr_1}')
	println('arr_2: ${arr_2}')

	ptr_1 := unsafe { &u8(&arr_1[0]) }
	ptr_2 := unsafe { &u8(&arr_2[0]) }

	hash_of_arr_1 := hash.wyhash_c(ptr_1, u64(arr_1.len) * sizeof(u64), 0)
	hash_of_arr_2 := hash.wyhash_c(ptr_2, u64(arr_2.len) * sizeof(u64), 0)

	println('hash_of_arr_1: ${hash_of_arr_1}')
	println('hash_of_arr_2: ${hash_of_arr_2}')
}
