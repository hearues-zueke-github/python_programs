import hash

struct TupleKeyArrayValueArray[K, V] {
	key []K
	val []V
}

struct MapKeyArrayValueArray[K, V] {
mut:
	map_key_val_intern map[u64][]TupleKeyArrayValueArray[K, V]
}

fn (mut map_key_val MapKeyArrayValueArray[K, V]) put_key_val(key []K, val []V) {
	ptr_u8 := unsafe { &u8(&key[0]) }
	hash_key_u64 := hash.wyhash_c(ptr_u8, u64(key.len) * sizeof(K), 0) & u64(0xFF)

	if hash_key_u64 in map_key_val.map_key_val_intern {
		// need to check, if the value already exists, if so, than update the values

		mut arr_tuple_key_val := unsafe { &(map_key_val.map_key_val_intern[hash_key_u64]) }
		for i, tuple_key_val in arr_tuple_key_val {
			if tuple_key_val.key == key {
				unsafe { arr_tuple_key_val[i] = TupleKeyArrayValueArray[K, V]{key: key, val: val} }
				return
			}
		}

		arr_tuple_key_val << TupleKeyArrayValueArray[K, V]{key: key, val: val}

		return
	}

	map_key_val.map_key_val_intern[hash_key_u64] = []TupleKeyArrayValueArray[K, V]{}
	map_key_val.map_key_val_intern[hash_key_u64] << TupleKeyArrayValueArray[K, V]{key: key, val: val}
}

fn (mut map_key_val MapKeyArrayValueArray[K, V]) get_val(key []K) ![]V {
	ptr_u8 := unsafe { &u8(&key[0]) }
	hash_key_u64 := hash.wyhash_c(ptr_u8, u64(key.len) * sizeof(K), 0) & u64(0xFF)

	if hash_key_u64 in map_key_val.map_key_val_intern {
		// need to check, if the value already exists, if so, return it, otherwise return error

		mut arr_tuple_key_val := unsafe { &(map_key_val.map_key_val_intern[hash_key_u64]) }
		for tuple_key_val in arr_tuple_key_val {
			if tuple_key_val.key == key {
				return tuple_key_val.val
			}
		}

		panic('key not found!')
	}

	panic('hash_key_u64 not found!')
}

fn (mut map_key_val MapKeyArrayValueArray[K, V]) delete_key(key []K) {
	ptr_u8 := unsafe { &u8(&key[0]) }
	hash_key_u64 := hash.wyhash_c(ptr_u8, u64(key.len) * sizeof(K), 0) & u64(0xFF)

	if hash_key_u64 in map_key_val.map_key_val_intern {
		// need to check, if the value already exists, if so, delete it, otherwise return error

		mut arr_tuple_key_val := unsafe { &(map_key_val.map_key_val_intern[hash_key_u64]) }
		for i, tuple_key_val in arr_tuple_key_val {
			if tuple_key_val.key == key {
				arr_tuple_key_val.delete(i)
				return
			}
		}

		panic('key not found!')
	}

	panic('hash_key_u64 not found!')
}

fn (mut map_key_val MapKeyArrayValueArray[K, V]) print() {
	println('Amount hashes: ${map_key_val.map_key_val_intern.keys().len}')

	for hash_key_u64 in map_key_val.map_key_val_intern.keys() {
		arr_tuple_key_val := unsafe { &(map_key_val.map_key_val_intern[hash_key_u64]) }
		println('hash_key_u64: 0x${hash_key_u64:016x}, amount elements: ${arr_tuple_key_val.len}')
		print('[')
		for i, tuple_key_val in arr_tuple_key_val {
			print('{')
			print('i: ${i}, key: ${tuple_key_val.key}, val: ${tuple_key_val.val}')
			print('}, ')
		}
		println(']')
	}
}
