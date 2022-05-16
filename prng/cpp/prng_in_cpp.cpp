#include <cassert>
#include <iostream>
#include <stdint.h>
#include <string>

#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include "OwnRND.h"

using std::cout;
using std::copy;
using std::endl;
using std::string;
using std::vector;

using fmt::format;
using fmt::print;

using OwnRND::RandomNumberDevice;

int main(int argc, char* argv[]) {
	print("Test!\n");

	const uint32_t a = 0x12345678;
	const uint8_t* b = (uint8_t*)&a;

	vector<uint8_t> c(b, b+4);

	print("a: 0x{:08X}\n", a);
	print("c: ");
	for (auto it = c.begin(); it != c.end(); ++it) {
		print("0x{:02X}, ", *it);
	}
	print("\n");

	RandomNumberDevice rnd = RandomNumberDevice(128);

	rnd.print_state();
	rnd.print_values();

	const size_t amount = 100000000;

	vector<uint64_t> vec;
	rnd.generate_new_values_uint64_t(vec, amount);

	// print("vec: {}\n", vec);

	vector<double> vec_double;
	rnd.generate_new_values_double(vec_double, amount);
	
	// print("vec_double: {}\n", vec_double);
	print("Hello Test!\n");

	// SHA256_CTX ctx;
	// uint8_t hash[32];
	// string hashStr = "";

	// SHA256Init(&ctx);
	// SHA256Update(&ctx, rnd.ptr_state_, rnd.amount_);
	// SHA256Final(&ctx, hash);

	// char s[3];
	// for (int i = 0; i < 32; i++) {
	// 	sprintf(s, "%02x", hash[i]);
	// 	hashStr += s;
	// }

	// print("hashStr: {}\n", hashStr);

  return 0;
}
