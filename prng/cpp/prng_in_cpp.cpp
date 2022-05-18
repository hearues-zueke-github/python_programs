#include <cassert>
#include <chrono>
#include <iostream>
#include <stdint.h>
#include <string>

#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include "OwnPRNG.h"

using std::cout;
using std::copy;
using std::endl;
using std::string;
using std::vector;

using fmt::format;
using fmt::print;

using OwnPRNG::RandomNumberDevice;

int main(int argc, char* argv[]) {
	const size_t amount = std::stoull(argv[1]);

	const auto time_1 = std::chrono::high_resolution_clock::now();
	RandomNumberDevice rnd = RandomNumberDevice(1024, {0x01});
	rnd.print_state();
	rnd.print_values();

	const auto time_2 = std::chrono::high_resolution_clock::now();
	vector<uint64_t> vec1;
	rnd.generate_new_values_uint64_t(vec1, 10);
	print("vec1: {}\n", vec1);
	vector<uint64_t> vec2;
	rnd.generate_new_values_uint64_t(vec2, 11);
	print("vec2: {}\n", vec2);
	vector<uint64_t> vec3;
	rnd.generate_new_values_uint64_t(vec3, 12);
	print("vec3: {}\n", vec3);
	vector<uint64_t> vec4;
	rnd.generate_new_values_uint64_t(vec4, 13);
	print("vec4: {}\n", vec4);
	vector<uint64_t> vec5;
	rnd.generate_new_values_uint64_t(vec5, 23);
	print("vec5: {}\n", vec5);
	vector<uint64_t> vec6;
	rnd.generate_new_values_uint64_t(vec6, 5);
	print("vec6: {}\n", vec6);
	vector<uint64_t> vec7;
	rnd.generate_new_values_uint64_t(vec7, 4);
	print("vec7: {}\n", vec7);
	vector<uint64_t> vec8;
	rnd.generate_new_values_uint64_t(vec8, 31);
	print("vec8: {}\n", vec8);

	vector<uint64_t> vec;
	rnd.generate_new_values_uint64_t(vec, amount);
	// print("vec: {}\n", vec);

	const auto time_3 = std::chrono::high_resolution_clock::now();
	vector<double> vec_double;
	rnd.generate_new_values_double(vec_double, amount);
	// print("vec_double: {}\n", vec_double);
	
	const auto time_4 = std::chrono::high_resolution_clock::now();
	
	const uint64_t duration_1 = std::chrono::duration_cast<std::chrono::nanoseconds>(time_2-time_1).count();
	const uint64_t duration_2 = std::chrono::duration_cast<std::chrono::nanoseconds>(time_3-time_2).count();
	const uint64_t duration_3 = std::chrono::duration_cast<std::chrono::nanoseconds>(time_4-time_3).count();

	print("duration_1: {}s\n", duration_1 / 1000000000.);
	print("duration_2: {}s\n", duration_2 / 1000000000.);
	print("duration_3: {}s\n", duration_3 / 1000000000.);

  return 0;
}
