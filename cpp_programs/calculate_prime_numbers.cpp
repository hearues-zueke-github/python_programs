#include <chrono>
#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"

using std::accumulate;
using std::fstream;
using std::vector;

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

using fmt::format;
using fmt::print;

using u64 = unsigned long long;

inline u64 sqrt_int(const u64 v_){
	u64 v1 = v_ / 2;
	u64 v2 = (v1 + v_ / v1) / 2;

	for (u64 i = 1; i < 10; i += 1) {
		const u64 v3 = (v2 + v_ / v2) / 2;

		if (v1 == v3 || v2 == v3) {
			break;
		}

		v1 = v2;
		v2 = v3;
	}

	return v2;
}

int main(int argc, char* argv[]) {
	assert((argc >= 3) && "Missing at least one more argument!");

	vector<double> l_diff;

	const u64 max_p = std::stoi(argv[1]);
	const u64 amount = std::stoi(argv[2]);
	
	for (u64 i_round = 0; i_round < amount; ++i_round) {
		auto start = high_resolution_clock::now();
		
		vector<u64> l = {2, 3, 5};
		const vector<u64> l_jump = {4, 2};
		
		u64 i_jump = 0ull;
		u64 p = 7ull;

		// const u64 max_p = 1000000ull;

		while (p < max_p) {
			const u64 max_sqrt_p = sqrt_int(p) + 1;

			// is p a prime number? let's test this
			bool is_prime = true;
			for (u64 i = 0; l[i] < max_sqrt_p; i += 1) {
				if (p % l[i] == 0) {
					is_prime = false;
					break;
				}
			}

			if (is_prime) {
				l.push_back(p);
			}

			p += l_jump[i_jump];
			i_jump = (i_jump + 1) % 2;
		}

		auto finish = high_resolution_clock::now();

		double elapsed_time = ((double)duration_cast<nanoseconds>(finish-start).count()) / 1000000000.;
		l_diff.push_back(elapsed_time);

		if (i_round == 0) {
			fstream f(format("/tmp/primes_n_{}_cpp.txt", max_p), std::ios::out);

			for (auto v : l) {
				f << format("{},", v);
			}

			f.close();
		}
	}

	double average_time = accumulate(l_diff.begin(), l_diff.end(), 0.) / l_diff.size();

	print("l_diff: {}\n", l_diff);
	print("average_time: {}\n", average_time);
}
