#include <iostream>
#include <iomanip>
#include <vector>
#include <span>
#include <cstdint>
#include <ostream>
#include <cassert>
#include <ctime>
#include <map>
#include <set>
#include <string>
#include <cinttypes>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <thread>
#include <mutex>
#include <utility>
#include <functional>
#include <sqlite3.h>
#include <cstdio>

#include "cyclic_multi_factor_pow_seq.h"

using std::cout;
using std::endl;
using std::vector;
using std::span;
using std::ostream;
using std::map;
using std::set;
using std::string;
using std::ofstream;
using std::stringstream;
using std::iota;
using std::stable_sort;
using std::thread;
using std::mutex;
using std::move;
using std::ref;
namespace chrono = std::chrono;
namespace this_thread = std::this_thread;

ostream& operator<<(ostream& os, const vector<uint64_t>& obj) {
	std::ios oldState(nullptr);
	oldState.copyfmt(os);

	os << "[";
	for (auto i : obj) {
		os << "0x" << std::setfill('0') << std::setw(16) << std::right << std::hex << i << "UL, ";
	}
	os << "]";

	os.copyfmt(oldState);

	return os;
}

template <typename S>
ostream& operator<<(ostream& os, const vector<S>& obj) {
	os << "[";
	for (auto i : obj) {
		os << i << ", ";
	}
	os << "]";

	return os;
}
template ostream& operator<<(ostream& os, const vector<size_t>& obj);

ostream& operator<<(ostream& os, const map<string, string>& obj) {
	os << "{";
	for (map<string, string>::const_iterator it = obj.begin(); it != obj.end(); ++it) {
		os << "\"" << it->first << "\": \"" << it->second << "\", ";
	}
	os << "}";

	return os;
}

vector<string> split(string s, const string& del) {
	vector<string> tokens;
	int end = s.find(del); 

	while (end != -1) {
		tokens.push_back(s.substr(0, end));
		s.erase(s.begin(), s.begin() + end + 1);
		end = s.find(del);
	}

	tokens.push_back(s.substr(0, end));

	return tokens;
}

template <typename T>
struct pointee_less
{
	bool operator()(T* lhs, T* rhs) const
	{
		return (lhs && rhs) ? std::less<T>{}(*lhs, *rhs) : std::less<T*>{}(lhs, rhs);
	}
};

template <typename T>
using pointer_set = set<T*, pointee_less<T>>;

class RecordCycleFactors {
public:
	const int32_t factor_amount;
	const int32_t modulo;
	const int32_t cycle_len;
	const int32_t factor_len;
	const vector<int32_t> t_cycle;
	const vector<int32_t> t_factor;
	const int32_t amount_nonzero_factors;
	string status;
	const string dt;

	RecordCycleFactors(
		const int32_t factor_amount,
		const int32_t modulo,
		const int32_t cycle_len,
		const int32_t factor_len,
		const vector<int32_t>& t_cycle,
		const vector<int32_t>& t_factor,
		const int32_t amount_nonzero_factors,
		const string status,
		const string dt
	) :
		factor_amount(factor_amount), modulo(modulo), cycle_len(cycle_len), factor_len(factor_len),
		t_cycle(t_cycle), t_factor(t_factor),
		amount_nonzero_factors(amount_nonzero_factors), status(status), dt(dt)
	{}

	friend bool operator< (const RecordCycleFactors& left, const RecordCycleFactors& right);
	friend ostream& operator<<(ostream& os, const vector<RecordCycleFactors>& obj);
};

bool operator< (const RecordCycleFactors& left, const RecordCycleFactors& right) {
	if (left.factor_amount < right.factor_amount) {
		return true;
	} else if (left.factor_amount > right.factor_amount) {
		return false;
	}

	if (left.modulo < right.modulo) {
		return true;
	} else if (left.modulo > right.modulo) {
		return false;
	}

	if (left.factor_len < right.factor_len) {
		return true;
	} else if (left.factor_len > right.factor_len) {
		return false;
	}

	if (left.t_cycle < right.t_cycle) {
		return true;
	} else if (left.t_cycle > right.t_cycle) {
		return false;
	}

	return false;
}

ostream& operator<<(ostream& os, const RecordCycleFactors& obj) {
	std::ios oldState(nullptr);
	oldState.copyfmt(os);

	os << "RecordCycleFactors(";
	os << "factor_amount: " << obj.factor_amount << ", ";
	os << "modulo: " << obj.modulo << ", ";
	os << "cycle_len: " << obj.cycle_len << ", ";
	os << "factor_len: " << obj.factor_len << ", ";
	os << "t_cycle: " << obj.t_cycle << ", ";
	os << "t_factor: " << obj.t_factor << ", ";
	os << "amount_nonzero_factors: " << obj.amount_nonzero_factors << ", ";
	os << "status: \"" << obj.status << "\", ";
	os << "dt: \"" << obj.dt << "\"";
	os << ")";

	os.copyfmt(oldState);

	return os;
}

class RecordIdCycleFactors : public RecordCycleFactors {
public:
	const int32_t id;

	RecordIdCycleFactors(
		const int32_t id,
		const int32_t factor_amount,
		const int32_t modulo,
		const int32_t cycle_len,
		const int32_t factor_len,
		const vector<int32_t>& t_cycle,
		const vector<int32_t>& t_factor,
		const int32_t amount_nonzero_factors,
		const string status,
		const string dt
	) :
		id(id),
		RecordCycleFactors(factor_amount, modulo, cycle_len, factor_len, t_cycle, t_factor, amount_nonzero_factors, status, dt)
	{}

	friend ostream& operator<<(ostream& os, const vector<RecordCycleFactors>& obj);
};

ostream& operator<<(ostream& os, const RecordIdCycleFactors& obj) {
	std::ios oldState(nullptr);
	oldState.copyfmt(os);

	os << "RecordIdCycleFactors(";
	os << "id: " << obj.id << ", ";
	os << "factor_amount: " << obj.factor_amount << ", ";
	os << "modulo: " << obj.modulo << ", ";
	os << "cycle_len: " << obj.cycle_len << ", ";
	os << "factor_len: " << obj.factor_len << ", ";
	os << "t_cycle: " << obj.t_cycle << ", ";
	os << "t_factor: " << obj.t_factor << ", ";
	os << "amount_nonzero_factors: " << obj.amount_nonzero_factors << ", ";
	os << "status: \"" << obj.status << "\", ";
	os << "dt: \"" << obj.dt << "\"";
	os << ")";

	os.copyfmt(oldState);

	return os;
}

class RecordCycleManyRandom {
public:
	vector<uint64_t> values_a;
	vector<uint64_t> values_c;
	const int32_t tries_per_thread;
	const int32_t factor_amount;
	const int32_t modulo;
	const int32_t factor_len;

	RecordCycleManyRandom(
		const vector<uint64_t>& values_a,
		const vector<uint64_t>& values_c,
		const int32_t tries_per_thread,
		const int32_t factor_amount,
		const int32_t modulo,
		const int32_t factor_len
	) :
		values_a(values_a), values_c(values_c), tries_per_thread(tries_per_thread),
		factor_amount(factor_amount), modulo(modulo), factor_len(factor_len)
	{}

	RecordCycleManyRandom(
		const RecordCycleManyRandom& obj
	) :
		values_a(obj.values_a), values_c(obj.values_c), tries_per_thread(obj.tries_per_thread),
		factor_amount(obj.factor_amount), modulo(obj.modulo), factor_len(obj.factor_len)
	{}
};

class RecordCycleManyRandomAmountNonzero : public RecordCycleManyRandom {
public:
	const int32_t amount_nonzero_factors;

	RecordCycleManyRandomAmountNonzero(
		const RecordCycleManyRandom& record_cycle_many_random,
		const int32_t amount_nonzero_factors
	) :
		RecordCycleManyRandom(record_cycle_many_random),
		amount_nonzero_factors(amount_nonzero_factors)
	{}
};

class SimpleRandomNumberGenerator {
public:
	const uint64_t a_fix = 0x123456789abcdef1;
	const uint64_t c_fix = 0x56789abcdef11235;
	uint64_t x_fix;

	vector<uint64_t> values_a;
	vector<uint64_t> values_c;
	vector<uint64_t> values_x;
	const size_t amount_vals;
	size_t current_index;

	SimpleRandomNumberGenerator(const vector<uint64_t>& values_a, const vector<uint64_t>& values_c) :
			values_a(), values_c(), values_x(), amount_vals(values_a.size()), current_index(0) {
		assert(values_a.size() > 0);
		assert(values_a.size() == values_c.size());

		for (uint64_t val: values_a) {
			assert((val - 1) % 4 == 0);
		}

		for (uint64_t val: values_c) {
			assert((val - 1) % 2 == 0);
		}

		this->values_a = values_a;
		this->values_c = values_c;

		x_fix = 0;
		for (size_t i = 0; i < this->amount_vals; ++i) {
			x_fix = a_fix * x_fix + c_fix;
		}

		// prepare some values already
		this->values_x.resize(this->amount_vals);
		this->values_x.assign(this->values_x.size(), 0);

		for (size_t i = 0; i < this->amount_vals; ++i) {
			this->values_x[i] = (
				(this->values_a[i] ^ this->values_c[i]) +
				((this->values_a[i] >> 8) ^ (this->values_c[i] << 8) ^ 0x0f0f0f0f0f0f0f0fUl) + 
				((this->values_a[i] << 8) ^ (this->values_c[i] >> 8) ^ 0xf0f0f0f0f0f0f0f0Ul)
			);
		}

		for (size_t i = 0; i < this->amount_vals * 2; ++i) {
			const uint64_t a = this->values_a[(i + 0) % this->amount_vals];
			const uint64_t c = this->values_c[(i + 1) % this->amount_vals];
			const uint64_t x = this->values_x[(i + 2) % this->amount_vals];
			uint64_t& x2 = this->values_x[(i + 3) % this->amount_vals];
			x2 = x2 ^ (a * x + c);
		}
	}

	inline uint64_t getNextU64Val() {
		this->x_fix = this->a_fix * this->x_fix + this->c_fix;

		const size_t current_index = this->current_index;
		const uint64_t x = (
			(this->values_x[current_index] * this->values_a[current_index] + this->values_c[current_index]) ^
			this->x_fix
		);
		this->values_x[current_index] = x;
		this->current_index = (this->current_index + 1) % this->amount_vals;

		return x;
	}

	void calcNextRandomValNew(const size_t amount, vector<uint64_t>& values) {
		values.resize(amount);
		for (size_t i = 0; i < amount; ++i) {
			values[i] = getNextU64Val();
		}
	}

	void calcNextRandomValAdd(const size_t amount, vector<uint64_t>& values) {
		for (size_t i = 0; i < amount; ++i) {
			values.push_back(getNextU64Val());
		}
	}

	void calcNextRandomValModulo(const size_t amount, const uint64_t modulo, vector<int32_t>& values) {
		assert(modulo <= 0xFFFF);

		values.resize(amount);
		values.assign(values.size(), 0);

		uint64_t val = 0;
		const size_t amount_multi = amount;

		for (size_t i_count = 0; i_count < amount_multi;) {
			val = getNextU64Val();
			// switch (i_count % 3) {
			// case 0:
			// 	val = getNextU64Val();
			// 	break;
			// case 1:
			// 	val = getNextU64Val() >> 8 + getNextU64Val() << 8;
			// 	break;
			// case 2:
			// 	val = getNextU64Val() >> 16 + getNextU64Val() << 16 + getNextU64Val() << 24;
			// 	break;
			// }

			while (i_count < amount_multi && val > modulo) {
				const int32_t val_mod = val % modulo;
				const size_t idx = i_count % amount;
				values[idx] = (values[idx] + (int32_t)(val % modulo)) % modulo;
				val /= modulo;
				i_count += 1;
			}
		}
	}
};

inline int32_t do_fac_1_cycle_full_poly_pow_1(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*1 + 0] = x0;

		const int32_t x0_pow2 = (x0 * x0) % modulo;

		const int32_t next_x = (
			(x0 * factors[0]) % modulo +
			(1 * factors[1])
		) % modulo;

		x0 = next_x;

		const int32_t idx = x0;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_1_cycle_full_poly_pow_2(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*1 + 0] = x0;

		const int32_t x0_pow2 = (x0 * x0) % modulo;

		const int32_t next_x = (
			(x0_pow2 * factors[0]) % modulo +
			(x0 * factors[1]) % modulo +
			(1 * factors[2])
		) % modulo;

		x0 = next_x;

		const int32_t idx = x0;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_1_cycle_full_poly_pow_3(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*1 + 0] = x0;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;

		const int32_t next_x = (
			(x0_pow3 * factors[0]) % modulo +
			(x0_pow2 * factors[1]) % modulo +
			(x0 * factors[2]) % modulo +
			(1 * factors[3])
		) % modulo;

		x0 = next_x;

		const int32_t idx = x0;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_1_cycle_full_poly_pow_4(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*1 + 0] = x0;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x0_pow4 = (x0_pow3 * x0) % modulo;

		const int32_t next_x = (
			(x0_pow4 * factors[0]) % modulo +
			(x0_pow3 * factors[1]) % modulo +
			(x0_pow2 * factors[2]) % modulo +
			(x0 * factors[3]) % modulo +
			(1 * factors[4])
		) % modulo;

		x0 = next_x;

		const int32_t idx = x0;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_1_cycle_full_poly_pow_5(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*1 + 0] = x0;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x0_pow4 = (x0_pow3 * x0) % modulo;
		const int32_t x0_pow5 = (x0_pow4 * x0) % modulo;

		const int32_t next_x = (
			(x0_pow5 * factors[0]) % modulo +
			(x0_pow4 * factors[1]) % modulo +
			(x0_pow3 * factors[2]) % modulo +
			(x0_pow2 * factors[3]) % modulo +
			(x0 * factors[4]) % modulo +
			(1 * factors[5])
		) % modulo;

		x0 = next_x;

		const int32_t idx = x0;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_1_cycle_full_poly_pow_6(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*1 + 0] = x0;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x0_pow4 = (x0_pow3 * x0) % modulo;
		const int32_t x0_pow5 = (x0_pow4 * x0) % modulo;
		const int32_t x0_pow6 = (x0_pow5 * x0) % modulo;

		const int32_t next_x = (
			(x0_pow6 * factors[0]) % modulo +
			(x0_pow5 * factors[1]) % modulo +
			(x0_pow4 * factors[2]) % modulo +
			(x0_pow3 * factors[3]) % modulo +
			(x0_pow2 * factors[4]) % modulo +
			(x0 * factors[5]) % modulo +
			(1 * factors[6])
		) % modulo;

		x0 = next_x;

		const int32_t idx = x0;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_1_cycle_full_poly_pow_7(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*1 + 0] = x0;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x0_pow4 = (x0_pow3 * x0) % modulo;
		const int32_t x0_pow5 = (x0_pow4 * x0) % modulo;
		const int32_t x0_pow6 = (x0_pow5 * x0) % modulo;
		const int32_t x0_pow7 = (x0_pow6 * x0) % modulo;

		const int32_t next_x = (
			(x0_pow7 * factors[0]) % modulo +
			(x0_pow6 * factors[1]) % modulo +
			(x0_pow5 * factors[2]) % modulo +
			(x0_pow4 * factors[3]) % modulo +
			(x0_pow3 * factors[4]) % modulo +
			(x0_pow2 * factors[5]) % modulo +
			(x0 * factors[6]) % modulo +
			(1 * factors[7])
		) % modulo;

		x0 = next_x;

		const int32_t idx = x0;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_1_cycle_full_poly_pow_8(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*1 + 0] = x0;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x0_pow4 = (x0_pow3 * x0) % modulo;
		const int32_t x0_pow5 = (x0_pow4 * x0) % modulo;
		const int32_t x0_pow6 = (x0_pow5 * x0) % modulo;
		const int32_t x0_pow7 = (x0_pow6 * x0) % modulo;
		const int32_t x0_pow8 = (x0_pow7 * x0) % modulo;

		const int32_t next_x = (
			(x0_pow8 * factors[0]) % modulo +
			(x0_pow7 * factors[1]) % modulo +
			(x0_pow6 * factors[2]) % modulo +
			(x0_pow5 * factors[3]) % modulo +
			(x0_pow4 * factors[4]) % modulo +
			(x0_pow3 * factors[5]) % modulo +
			(x0_pow2 * factors[6]) % modulo +
			(x0 * factors[7]) % modulo +
			(1 * factors[8])
		) % modulo;

		x0 = next_x;

		const int32_t idx = x0;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_1_cycle_full_poly(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	values.assign(values.size(), 0);
	cycle_checker.assign(cycle_checker.size(), 0);

	switch (factors.size()) {
		case 2: return do_fac_1_cycle_full_poly_pow_1(modulo, cycle_len, factors, values, cycle_checker);
		case 3: return do_fac_1_cycle_full_poly_pow_2(modulo, cycle_len, factors, values, cycle_checker);
		case 4: return do_fac_1_cycle_full_poly_pow_3(modulo, cycle_len, factors, values, cycle_checker);
		case 5: return do_fac_1_cycle_full_poly_pow_4(modulo, cycle_len, factors, values, cycle_checker);
		case 6: return do_fac_1_cycle_full_poly_pow_5(modulo, cycle_len, factors, values, cycle_checker);
		case 7: return do_fac_1_cycle_full_poly_pow_6(modulo, cycle_len, factors, values, cycle_checker);
		case 8: return do_fac_1_cycle_full_poly_pow_7(modulo, cycle_len, factors, values, cycle_checker);
		case 9: return do_fac_1_cycle_full_poly_pow_8(modulo, cycle_len, factors, values, cycle_checker);
		default:
			assert(false);
			break;
	}
}

inline int32_t do_fac_2_cycle_full_poly_pow_1(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	int32_t x1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*2 + 0] = x0;
		values[i*2 + 1] = x1;

		const int32_t next_x = (
			(x0 * x1 * factors[0]) % modulo +

			(1 * x1 * factors[1]) % modulo +
			(x0 * 1 * factors[2]) % modulo +


			(1 * 1 * factors[3])
		) % modulo;

		x0 = x1;
		x1 = next_x;

		const int32_t idx = x0 * modulo + x1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_2_cycle_full_poly_pow_2(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	int32_t x1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*2 + 0] = x0;
		values[i*2 + 1] = x1;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x1_pow2 = (x1 * x1) % modulo;

		const int32_t next_x = (
			(x0_pow2 * x1_pow2 * factors[0]) % modulo +

			(x0 * x1_pow2 * factors[1]) % modulo +
			(x0_pow2 * x1 * factors[2]) % modulo +

			(1 * x1_pow2 * factors[3]) % modulo +
			(x0_pow2 * 1 * factors[4]) % modulo +


			(x0 * x1 * factors[5]) % modulo +

			(1 * x1 * factors[6]) % modulo +
			(x0 * 1 * factors[7]) % modulo +


			(1 * 1 * factors[8])
		) % modulo;

		x0 = x1;
		x1 = next_x;

		const int32_t idx = x0 * modulo + x1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_2_cycle_full_poly_pow_3(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	int32_t x1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*2 + 0] = x0;
		values[i*2 + 1] = x1;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x1_pow2 = (x1 * x1) % modulo;

		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x1_pow3 = (x1_pow2 * x1) % modulo;

		const int32_t next_x = (
			(x0_pow3 * x1_pow3 * factors[0]) % modulo +

			(x0_pow2 * x1_pow3 * factors[1]) % modulo +
			(x0_pow3 * x1_pow2 * factors[2]) % modulo +

			(x0 * x1_pow3 * factors[3]) % modulo +
			(x0_pow3 * x1 * factors[4]) % modulo +

			(1 * x1_pow3 * factors[5]) % modulo +
			(x0_pow3 * 1 * factors[6]) % modulo +


			(x0_pow2 * x1_pow2 * factors[7]) % modulo +

			(x0 * x1_pow2 * factors[8]) % modulo +
			(x0_pow2 * x1 * factors[9]) % modulo +

			(1 * x1_pow2 * factors[10]) % modulo +
			(x0_pow2 * 1 * factors[11]) % modulo +


			(x0 * x1 * factors[12]) % modulo +

			(1 * x1 * factors[13]) % modulo +
			(x0 * 1 * factors[14]) % modulo +


			(1 * 1 * factors[15])
		) % modulo;

		x0 = x1;
		x1 = next_x;

		const int32_t idx = x0 * modulo + x1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_2_cycle_full_poly_pow_4(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	int32_t x1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*2 + 0] = x0;
		values[i*2 + 1] = x1;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x1_pow2 = (x1 * x1) % modulo;

		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x1_pow3 = (x1_pow2 * x1) % modulo;

		const int32_t x0_pow4 = (x0_pow3 * x0) % modulo;
		const int32_t x1_pow4 = (x1_pow3 * x1) % modulo;

		const int32_t next_x = (
			(x0_pow4 * x1_pow4 * factors[0]) % modulo +

			(x0_pow3 * x1_pow4 * factors[1]) % modulo +
			(x0_pow4 * x1_pow3 * factors[2]) % modulo +

			(x0_pow2 * x1_pow4 * factors[3]) % modulo +
			(x0_pow4 * x1_pow2 * factors[4]) % modulo +

			(x0 * x1_pow4 * factors[5]) % modulo +
			(x0_pow4 * x1 * factors[6]) % modulo +

			(1 * x1_pow4 * factors[7]) % modulo +
			(x0_pow4 * 1 * factors[8]) % modulo +


			(x0_pow3 * x1_pow3 * factors[9]) % modulo +

			(x0_pow2 * x1_pow3 * factors[10]) % modulo +
			(x0_pow3 * x1_pow2 * factors[11]) % modulo +

			(x0 * x1_pow3 * factors[12]) % modulo +
			(x0_pow3 * x1 * factors[13]) % modulo +

			(1 * x1_pow3 * factors[14]) % modulo +
			(x0_pow3 * 1 * factors[15]) % modulo +


			(x0_pow2 * x1_pow2 * factors[16]) % modulo +

			(x0 * x1_pow2 * factors[17]) % modulo +
			(x0_pow2 * x1 * factors[18]) % modulo +

			(1 * x1_pow2 * factors[19]) % modulo +
			(x0_pow2 * 1 * factors[20]) % modulo +


			(x0 * x1 * factors[21]) % modulo +

			(1 * x1 * factors[22]) % modulo +
			(x0 * 1 * factors[23]) % modulo +


			(1 * 1 * factors[24])
		) % modulo;

		x0 = x1;
		x1 = next_x;

		const int32_t idx = x0 * modulo + x1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_2_cycle_full_poly_pow_5(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	int32_t x1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*2 + 0] = x0;
		values[i*2 + 1] = x1;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x1_pow2 = (x1 * x1) % modulo;

		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x1_pow3 = (x1_pow2 * x1) % modulo;

		const int32_t x0_pow4 = (x0_pow3 * x0) % modulo;
		const int32_t x1_pow4 = (x1_pow3 * x1) % modulo;

		const int32_t x0_pow5 = (x0_pow4 * x0) % modulo;
		const int32_t x1_pow5 = (x1_pow4 * x1) % modulo;

		const int32_t next_x = (
			(x0_pow5 * x1_pow5 * factors[0]) % modulo +

			(x0_pow4 * x1_pow5 * factors[1]) % modulo +
			(x0_pow5 * x1_pow4 * factors[2]) % modulo +

			(x0_pow3 * x1_pow5 * factors[3]) % modulo +
			(x0_pow5 * x1_pow3 * factors[4]) % modulo +

			(x0_pow2 * x1_pow5 * factors[5]) % modulo +
			(x0_pow5 * x1_pow2 * factors[6]) % modulo +

			(x0 * x1_pow5 * factors[7]) % modulo +
			(x0_pow5 * x1 * factors[8]) % modulo +

			(1 * x1_pow5 * factors[9]) % modulo +
			(x0_pow5 * 1 * factors[10]) % modulo +


			(x0_pow4 * x1_pow4 * factors[11]) % modulo +

			(x0_pow3 * x1_pow4 * factors[12]) % modulo +
			(x0_pow4 * x1_pow3 * factors[13]) % modulo +

			(x0_pow2 * x1_pow4 * factors[14]) % modulo +
			(x0_pow4 * x1_pow2 * factors[15]) % modulo +

			(x0 * x1_pow4 * factors[16]) % modulo +
			(x0_pow4 * x1 * factors[17]) % modulo +

			(1 * x1_pow4 * factors[18]) % modulo +
			(x0_pow4 * 1 * factors[19]) % modulo +


			(x0_pow3 * x1_pow3 * factors[20]) % modulo +

			(x0_pow2 * x1_pow3 * factors[21]) % modulo +
			(x0_pow3 * x1_pow2 * factors[22]) % modulo +

			(x0 * x1_pow3 * factors[23]) % modulo +
			(x0_pow3 * x1 * factors[24]) % modulo +

			(1 * x1_pow3 * factors[25]) % modulo +
			(x0_pow3 * 1 * factors[26]) % modulo +


			(x0_pow2 * x1_pow2 * factors[27]) % modulo +

			(x0 * x1_pow2 * factors[28]) % modulo +
			(x0_pow2 * x1 * factors[29]) % modulo +

			(1 * x1_pow2 * factors[30]) % modulo +
			(x0_pow2 * 1 * factors[31]) % modulo +


			(x0 * x1 * factors[32]) % modulo +

			(1 * x1 * factors[33]) % modulo +
			(x0 * 1 * factors[34]) % modulo +


			(1 * 1 * factors[35])
		) % modulo;

		x0 = x1;
		x1 = next_x;

		const int32_t idx = x0 * modulo + x1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_2_cycle_full_poly_pow_6(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	int32_t x1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*2 + 0] = x0;
		values[i*2 + 1] = x1;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x1_pow2 = (x1 * x1) % modulo;

		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x1_pow3 = (x1_pow2 * x1) % modulo;

		const int32_t x0_pow4 = (x0_pow3 * x0) % modulo;
		const int32_t x1_pow4 = (x1_pow3 * x1) % modulo;

		const int32_t x0_pow5 = (x0_pow4 * x0) % modulo;
		const int32_t x1_pow5 = (x1_pow4 * x1) % modulo;

		const int32_t x0_pow6 = (x0_pow5 * x0) % modulo;
		const int32_t x1_pow6 = (x1_pow5 * x1) % modulo;

		const int32_t next_x = (
			(x0_pow6 * x1_pow6 * factors[0]) % modulo +

			(x0_pow5 * x1_pow6 * factors[1]) % modulo +
			(x0_pow6 * x1_pow5 * factors[2]) % modulo +

			(x0_pow4 * x1_pow6 * factors[3]) % modulo +
			(x0_pow6 * x1_pow4 * factors[4]) % modulo +

			(x0_pow3 * x1_pow6 * factors[5]) % modulo +
			(x0_pow6 * x1_pow3 * factors[6]) % modulo +

			(x0_pow2 * x1_pow6 * factors[7]) % modulo +
			(x0_pow6 * x1_pow2 * factors[8]) % modulo +

			(x0 * x1_pow6 * factors[9]) % modulo +
			(x0_pow6 * x1 * factors[10]) % modulo +

			(1 * x1_pow6 * factors[11]) % modulo +
			(x0_pow6 * 1 * factors[12]) % modulo +


			(x0_pow5 * x1_pow5 * factors[13]) % modulo +

			(x0_pow4 * x1_pow5 * factors[14]) % modulo +
			(x0_pow5 * x1_pow4 * factors[15]) % modulo +

			(x0_pow3 * x1_pow5 * factors[16]) % modulo +
			(x0_pow5 * x1_pow3 * factors[17]) % modulo +

			(x0_pow2 * x1_pow5 * factors[18]) % modulo +
			(x0_pow5 * x1_pow2 * factors[19]) % modulo +

			(x0 * x1_pow5 * factors[20]) % modulo +
			(x0_pow5 * x1 * factors[21]) % modulo +

			(1 * x1_pow5 * factors[22]) % modulo +
			(x0_pow5 * 1 * factors[23]) % modulo +


			(x0_pow4 * x1_pow4 * factors[24]) % modulo +

			(x0_pow3 * x1_pow4 * factors[25]) % modulo +
			(x0_pow4 * x1_pow3 * factors[26]) % modulo +

			(x0_pow2 * x1_pow4 * factors[27]) % modulo +
			(x0_pow4 * x1_pow2 * factors[28]) % modulo +

			(x0 * x1_pow4 * factors[29]) % modulo +
			(x0_pow4 * x1 * factors[30]) % modulo +

			(1 * x1_pow4 * factors[31]) % modulo +
			(x0_pow4 * 1 * factors[32]) % modulo +


			(x0_pow3 * x1_pow3 * factors[33]) % modulo +

			(x0_pow2 * x1_pow3 * factors[34]) % modulo +
			(x0_pow3 * x1_pow2 * factors[35]) % modulo +

			(x0 * x1_pow3 * factors[36]) % modulo +
			(x0_pow3 * x1 * factors[37]) % modulo +

			(1 * x1_pow3 * factors[38]) % modulo +
			(x0_pow3 * 1 * factors[39]) % modulo +


			(x0_pow2 * x1_pow2 * factors[40]) % modulo +

			(x0 * x1_pow2 * factors[41]) % modulo +
			(x0_pow2 * x1 * factors[42]) % modulo +

			(1 * x1_pow2 * factors[43]) % modulo +
			(x0_pow2 * 1 * factors[44]) % modulo +


			(x0 * x1 * factors[45]) % modulo +

			(1 * x1 * factors[46]) % modulo +
			(x0 * 1 * factors[47]) % modulo +


			(1 * 1 * factors[48])
		) % modulo;

		x0 = x1;
		x1 = next_x;

		const int32_t idx = x0 * modulo + x1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_2_cycle_full_poly_pow_7(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	int32_t x1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*2 + 0] = x0;
		values[i*2 + 1] = x1;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x1_pow2 = (x1 * x1) % modulo;

		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x1_pow3 = (x1_pow2 * x1) % modulo;

		const int32_t x0_pow4 = (x0_pow3 * x0) % modulo;
		const int32_t x1_pow4 = (x1_pow3 * x1) % modulo;

		const int32_t x0_pow5 = (x0_pow4 * x0) % modulo;
		const int32_t x1_pow5 = (x1_pow4 * x1) % modulo;

		const int32_t x0_pow6 = (x0_pow5 * x0) % modulo;
		const int32_t x1_pow6 = (x1_pow5 * x1) % modulo;

		const int32_t x0_pow7 = (x0_pow6 * x0) % modulo;
		const int32_t x1_pow7 = (x1_pow6 * x1) % modulo;

		const int32_t next_x = (
			(x0_pow7 * x1_pow7 * factors[0]) % modulo +

			(x0_pow6 * x1_pow7 * factors[1]) % modulo +
			(x0_pow7 * x1_pow6 * factors[2]) % modulo +

			(x0_pow5 * x1_pow7 * factors[3]) % modulo +
			(x0_pow7 * x1_pow5 * factors[4]) % modulo +

			(x0_pow4 * x1_pow7 * factors[5]) % modulo +
			(x0_pow7 * x1_pow4 * factors[6]) % modulo +

			(x0_pow3 * x1_pow7 * factors[7]) % modulo +
			(x0_pow7 * x1_pow3 * factors[8]) % modulo +

			(x0_pow2 * x1_pow7 * factors[9]) % modulo +
			(x0_pow7 * x1_pow2 * factors[10]) % modulo +

			(x0 * x1_pow7 * factors[11]) % modulo +
			(x0_pow7 * x1 * factors[12]) % modulo +

			(1 * x1_pow7 * factors[13]) % modulo +
			(x0_pow7 * 1 * factors[14]) % modulo +


			(x0_pow6 * x1_pow6 * factors[15]) % modulo +

			(x0_pow5 * x1_pow6 * factors[16]) % modulo +
			(x0_pow6 * x1_pow5 * factors[17]) % modulo +

			(x0_pow4 * x1_pow6 * factors[18]) % modulo +
			(x0_pow6 * x1_pow4 * factors[19]) % modulo +

			(x0_pow3 * x1_pow6 * factors[20]) % modulo +
			(x0_pow6 * x1_pow3 * factors[21]) % modulo +

			(x0_pow2 * x1_pow6 * factors[22]) % modulo +
			(x0_pow6 * x1_pow2 * factors[23]) % modulo +

			(x0 * x1_pow6 * factors[24]) % modulo +
			(x0_pow6 * x1 * factors[25]) % modulo +

			(1 * x1_pow6 * factors[26]) % modulo +
			(x0_pow6 * 1 * factors[27]) % modulo +


			(x0_pow5 * x1_pow5 * factors[28]) % modulo +

			(x0_pow4 * x1_pow5 * factors[29]) % modulo +
			(x0_pow5 * x1_pow4 * factors[30]) % modulo +

			(x0_pow3 * x1_pow5 * factors[31]) % modulo +
			(x0_pow5 * x1_pow3 * factors[32]) % modulo +

			(x0_pow2 * x1_pow5 * factors[33]) % modulo +
			(x0_pow5 * x1_pow2 * factors[34]) % modulo +

			(x0 * x1_pow5 * factors[35]) % modulo +
			(x0_pow5 * x1 * factors[36]) % modulo +

			(1 * x1_pow5 * factors[37]) % modulo +
			(x0_pow5 * 1 * factors[38]) % modulo +


			(x0_pow4 * x1_pow4 * factors[39]) % modulo +

			(x0_pow3 * x1_pow4 * factors[40]) % modulo +
			(x0_pow4 * x1_pow3 * factors[41]) % modulo +

			(x0_pow2 * x1_pow4 * factors[42]) % modulo +
			(x0_pow4 * x1_pow2 * factors[43]) % modulo +

			(x0 * x1_pow4 * factors[44]) % modulo +
			(x0_pow4 * x1 * factors[45]) % modulo +

			(1 * x1_pow4 * factors[46]) % modulo +
			(x0_pow4 * 1 * factors[47]) % modulo +


			(x0_pow3 * x1_pow3 * factors[48]) % modulo +

			(x0_pow2 * x1_pow3 * factors[49]) % modulo +
			(x0_pow3 * x1_pow2 * factors[50]) % modulo +

			(x0 * x1_pow3 * factors[51]) % modulo +
			(x0_pow3 * x1 * factors[52]) % modulo +

			(1 * x1_pow3 * factors[53]) % modulo +
			(x0_pow3 * 1 * factors[54]) % modulo +


			(x0_pow2 * x1_pow2 * factors[55]) % modulo +

			(x0 * x1_pow2 * factors[56]) % modulo +
			(x0_pow2 * x1 * factors[57]) % modulo +

			(1 * x1_pow2 * factors[58]) % modulo +
			(x0_pow2 * 1 * factors[59]) % modulo +


			(x0 * x1 * factors[60]) % modulo +

			(1 * x1 * factors[61]) % modulo +
			(x0 * 1 * factors[62]) % modulo +


			(1 * 1 * factors[63])
		) % modulo;

		x0 = x1;
		x1 = next_x;

		const int32_t idx = x0 * modulo + x1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_2_cycle_full_poly_pow_8(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	int32_t x0 = 0;
	int32_t x1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*2 + 0] = x0;
		values[i*2 + 1] = x1;

		const int32_t x0_pow2 = (x0 * x0) % modulo;
		const int32_t x1_pow2 = (x1 * x1) % modulo;

		const int32_t x0_pow3 = (x0_pow2 * x0) % modulo;
		const int32_t x1_pow3 = (x1_pow2 * x1) % modulo;

		const int32_t x0_pow4 = (x0_pow3 * x0) % modulo;
		const int32_t x1_pow4 = (x1_pow3 * x1) % modulo;

		const int32_t x0_pow5 = (x0_pow4 * x0) % modulo;
		const int32_t x1_pow5 = (x1_pow4 * x1) % modulo;

		const int32_t x0_pow6 = (x0_pow5 * x0) % modulo;
		const int32_t x1_pow6 = (x1_pow5 * x1) % modulo;

		const int32_t x0_pow7 = (x0_pow6 * x0) % modulo;
		const int32_t x1_pow7 = (x1_pow6 * x1) % modulo;

		const int32_t x0_pow8 = (x0_pow7 * x0) % modulo;
		const int32_t x1_pow8 = (x1_pow7 * x1) % modulo;

		const int32_t next_x = (
			(x0_pow8 * x1_pow8 * factors[0]) % modulo +

			(x0_pow7 * x1_pow8 * factors[1]) % modulo +
			(x0_pow8 * x1_pow7 * factors[2]) % modulo +

			(x0_pow6 * x1_pow8 * factors[3]) % modulo +
			(x0_pow8 * x1_pow6 * factors[4]) % modulo +

			(x0_pow5 * x1_pow8 * factors[5]) % modulo +
			(x0_pow8 * x1_pow5 * factors[6]) % modulo +

			(x0_pow4 * x1_pow8 * factors[7]) % modulo +
			(x0_pow8 * x1_pow4 * factors[8]) % modulo +

			(x0_pow3 * x1_pow8 * factors[9]) % modulo +
			(x0_pow8 * x1_pow3 * factors[10]) % modulo +

			(x0_pow2 * x1_pow8 * factors[11]) % modulo +
			(x0_pow8 * x1_pow2 * factors[12]) % modulo +

			(x0 * x1_pow8 * factors[13]) % modulo +
			(x0_pow8 * x1 * factors[14]) % modulo +

			(1 * x1_pow8 * factors[15]) % modulo +
			(x0_pow8 * 1 * factors[16]) % modulo +


			(x0_pow7 * x1_pow7 * factors[17]) % modulo +

			(x0_pow6 * x1_pow7 * factors[18]) % modulo +
			(x0_pow7 * x1_pow6 * factors[19]) % modulo +

			(x0_pow5 * x1_pow7 * factors[20]) % modulo +
			(x0_pow7 * x1_pow5 * factors[21]) % modulo +

			(x0_pow4 * x1_pow7 * factors[22]) % modulo +
			(x0_pow7 * x1_pow4 * factors[23]) % modulo +

			(x0_pow3 * x1_pow7 * factors[24]) % modulo +
			(x0_pow7 * x1_pow3 * factors[25]) % modulo +

			(x0_pow2 * x1_pow7 * factors[26]) % modulo +
			(x0_pow7 * x1_pow2 * factors[27]) % modulo +

			(x0 * x1_pow7 * factors[28]) % modulo +
			(x0_pow7 * x1 * factors[29]) % modulo +

			(1 * x1_pow7 * factors[30]) % modulo +
			(x0_pow7 * 1 * factors[31]) % modulo +


			(x0_pow6 * x1_pow6 * factors[32]) % modulo +

			(x0_pow5 * x1_pow6 * factors[33]) % modulo +
			(x0_pow6 * x1_pow5 * factors[34]) % modulo +

			(x0_pow4 * x1_pow6 * factors[35]) % modulo +
			(x0_pow6 * x1_pow4 * factors[36]) % modulo +

			(x0_pow3 * x1_pow6 * factors[37]) % modulo +
			(x0_pow6 * x1_pow3 * factors[38]) % modulo +

			(x0_pow2 * x1_pow6 * factors[39]) % modulo +
			(x0_pow6 * x1_pow2 * factors[40]) % modulo +

			(x0 * x1_pow6 * factors[41]) % modulo +
			(x0_pow6 * x1 * factors[42]) % modulo +

			(1 * x1_pow6 * factors[43]) % modulo +
			(x0_pow6 * 1 * factors[44]) % modulo +


			(x0_pow5 * x1_pow5 * factors[45]) % modulo +

			(x0_pow4 * x1_pow5 * factors[46]) % modulo +
			(x0_pow5 * x1_pow4 * factors[47]) % modulo +

			(x0_pow3 * x1_pow5 * factors[48]) % modulo +
			(x0_pow5 * x1_pow3 * factors[49]) % modulo +

			(x0_pow2 * x1_pow5 * factors[50]) % modulo +
			(x0_pow5 * x1_pow2 * factors[51]) % modulo +

			(x0 * x1_pow5 * factors[52]) % modulo +
			(x0_pow5 * x1 * factors[53]) % modulo +

			(1 * x1_pow5 * factors[54]) % modulo +
			(x0_pow5 * 1 * factors[55]) % modulo +


			(x0_pow4 * x1_pow4 * factors[56]) % modulo +

			(x0_pow3 * x1_pow4 * factors[57]) % modulo +
			(x0_pow4 * x1_pow3 * factors[58]) % modulo +

			(x0_pow2 * x1_pow4 * factors[59]) % modulo +
			(x0_pow4 * x1_pow2 * factors[60]) % modulo +

			(x0 * x1_pow4 * factors[61]) % modulo +
			(x0_pow4 * x1 * factors[62]) % modulo +

			(1 * x1_pow4 * factors[63]) % modulo +
			(x0_pow4 * 1 * factors[64]) % modulo +


			(x0_pow3 * x1_pow3 * factors[65]) % modulo +

			(x0_pow2 * x1_pow3 * factors[66]) % modulo +
			(x0_pow3 * x1_pow2 * factors[67]) % modulo +

			(x0 * x1_pow3 * factors[68]) % modulo +
			(x0_pow3 * x1 * factors[69]) % modulo +

			(1 * x1_pow3 * factors[70]) % modulo +
			(x0_pow3 * 1 * factors[71]) % modulo +


			(x0_pow2 * x1_pow2 * factors[72]) % modulo +

			(x0 * x1_pow2 * factors[73]) % modulo +
			(x0_pow2 * x1 * factors[74]) % modulo +

			(1 * x1_pow2 * factors[75]) % modulo +
			(x0_pow2 * 1 * factors[76]) % modulo +


			(x0 * x1 * factors[77]) % modulo +

			(1 * x1 * factors[78]) % modulo +
			(x0 * 1 * factors[79]) % modulo +


			(1 * 1 * factors[80])
		) % modulo;

		x0 = x1;
		x1 = next_x;

		const int32_t idx = x0 * modulo + x1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_2_cycle_full_poly(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	values.assign(values.size(), 0);
	cycle_checker.assign(cycle_checker.size(), 0);

	switch (factors.size()) {
		case 4: return do_fac_2_cycle_full_poly_pow_1(modulo, cycle_len, factors, values, cycle_checker);
		case 9: return do_fac_2_cycle_full_poly_pow_2(modulo, cycle_len, factors, values, cycle_checker);
		case 16: return do_fac_2_cycle_full_poly_pow_3(modulo, cycle_len, factors, values, cycle_checker);
		case 25: return do_fac_2_cycle_full_poly_pow_4(modulo, cycle_len, factors, values, cycle_checker);
		case 36: return do_fac_2_cycle_full_poly_pow_5(modulo, cycle_len, factors, values, cycle_checker);
		case 49: return do_fac_2_cycle_full_poly_pow_6(modulo, cycle_len, factors, values, cycle_checker);
		case 64: return do_fac_2_cycle_full_poly_pow_7(modulo, cycle_len, factors, values, cycle_checker);
		case 81: return do_fac_2_cycle_full_poly_pow_8(modulo, cycle_len, factors, values, cycle_checker);
		default:
			assert(false);
			break;
	}
}

inline int32_t do_fac_3_cycle_full_poly_pow_1(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	const int32_t x0_pow0 = 1;
	const int32_t x1_pow0 = 1;
	const int32_t x2_pow0 = 1;
	
	int32_t x0_pow1 = 0;
	int32_t x1_pow1 = 0;
	int32_t x2_pow1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*3 + 0] = x0_pow1;
		values[i*3 + 1] = x1_pow1;
		values[i*3 + 2] = x2_pow1;

		const int32_t next_x = (
			(x0_pow1 * x1_pow1 * x2_pow1 * factors[0]) % modulo +
			
			(x0_pow0 * x1_pow1 * x2_pow1 * factors[1]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow1 * factors[2]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow0 * factors[3]) % modulo +

			(x0_pow0 * x1_pow0 * x2_pow1 * factors[4]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow0 * factors[5]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow0 * factors[6]) % modulo +


			(x0_pow0 * x1_pow0 * x2_pow0 * factors[7])
		) % modulo;

		x0_pow1 = x1_pow1;
		x1_pow1 = x2_pow1;
		x2_pow1 = next_x;

		const int32_t idx = x0_pow1 * modulo * modulo + x1_pow1 * modulo + x2_pow1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_3_cycle_full_poly_pow_2(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	const int32_t x0_pow0 = 1;
	const int32_t x1_pow0 = 1;
	const int32_t x2_pow0 = 1;

	int32_t x0_pow1 = 0;
	int32_t x1_pow1 = 0;
	int32_t x2_pow1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*3 + 0] = x0_pow1;
		values[i*3 + 1] = x1_pow1;
		values[i*3 + 2] = x2_pow1;

		const int32_t x0_pow2 = (x0_pow1 * x0_pow1) % modulo;
		const int32_t x1_pow2 = (x1_pow1 * x1_pow1) % modulo;
		const int32_t x2_pow2 = (x2_pow1 * x2_pow1) % modulo;

		const int32_t next_x = (
			(x0_pow2 * x1_pow2 * x2_pow2 * factors[0]) % modulo +

			(x2_pow1 * x1_pow2 * x2_pow2 * factors[1]) % modulo +
			(x0_pow2 * x2_pow1 * x2_pow2 * factors[2]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow1 * factors[3]) % modulo +

			(x0_pow0 * x1_pow2 * x2_pow2 * factors[4]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow2 * factors[5]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow0 * factors[6]) % modulo +
			
			(x0_pow1 * x1_pow1 * x2_pow2 * factors[7]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow1 * factors[8]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow1 * factors[9]) % modulo +

			(x0_pow0 * x1_pow1 * x2_pow2 * factors[10]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow2 * factors[11]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow1 * factors[12]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow0 * factors[13]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow1 * factors[14]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow0 * factors[15]) % modulo +

			(x0_pow0 * x1_pow0 * x2_pow2 * factors[16]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow0 * factors[17]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow0 * factors[18]) % modulo +


			(x0_pow1 * x1_pow1 * x2_pow1 * factors[19]) % modulo +
			
			(x0_pow0 * x1_pow1 * x2_pow1 * factors[20]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow1 * factors[21]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow0 * factors[22]) % modulo +

			(x0_pow0 * x1_pow0 * x2_pow1 * factors[23]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow0 * factors[24]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow0 * factors[25]) % modulo +


			(x0_pow0 * x1_pow0 * x2_pow0 * factors[26])
		) % modulo;

		x0_pow1 = x1_pow1;
		x1_pow1 = x2_pow1;
		x2_pow1 = next_x;

		const int32_t idx = x0_pow1 * modulo * modulo + x1_pow1 * modulo + x2_pow1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_3_cycle_full_poly_pow_3(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	const int32_t x0_pow0 = 1;
	const int32_t x1_pow0 = 1;
	const int32_t x2_pow0 = 1;

	int32_t x0_pow1 = 0;
	int32_t x1_pow1 = 0;
	int32_t x2_pow1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*3 + 0] = x0_pow1;
		values[i*3 + 1] = x1_pow1;
		values[i*3 + 2] = x2_pow1;

		const int32_t x0_pow2 = (x0_pow1 * x0_pow1) % modulo;
		const int32_t x1_pow2 = (x1_pow1 * x1_pow1) % modulo;
		const int32_t x2_pow2 = (x2_pow1 * x2_pow1) % modulo;

		const int32_t x0_pow3 = (x0_pow2 * x0_pow1) % modulo;
		const int32_t x1_pow3 = (x1_pow2 * x1_pow1) % modulo;
		const int32_t x2_pow3 = (x2_pow2 * x2_pow1) % modulo;

		const int32_t next_x = (
			(x0_pow3 * x1_pow3 * x2_pow3 * factors[0]) % modulo +

			(x2_pow2 * x1_pow3 * x2_pow3 * factors[1]) % modulo +
			(x0_pow3 * x2_pow2 * x2_pow3 * factors[2]) % modulo +
			(x0_pow3 * x1_pow3 * x2_pow2 * factors[3]) % modulo +

			(x2_pow1 * x1_pow3 * x2_pow3 * factors[4]) % modulo +
			(x0_pow3 * x2_pow1 * x2_pow3 * factors[5]) % modulo +
			(x0_pow3 * x1_pow3 * x2_pow1 * factors[6]) % modulo +

			(x0_pow0 * x1_pow3 * x2_pow3 * factors[7]) % modulo +
			(x0_pow3 * x1_pow0 * x2_pow3 * factors[8]) % modulo +
			(x0_pow3 * x1_pow3 * x2_pow0 * factors[9]) % modulo +
			
			(x0_pow2 * x1_pow2 * x2_pow3 * factors[10]) % modulo +
			(x0_pow2 * x1_pow3 * x2_pow2 * factors[11]) % modulo +
			(x0_pow3 * x1_pow2 * x2_pow2 * factors[12]) % modulo +

			(x0_pow1 * x1_pow2 * x2_pow3 * factors[13]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow3 * factors[14]) % modulo +
			(x0_pow1 * x1_pow3 * x2_pow2 * factors[15]) % modulo +
			(x0_pow2 * x1_pow3 * x2_pow1 * factors[16]) % modulo +
			(x0_pow3 * x1_pow1 * x2_pow2 * factors[17]) % modulo +
			(x0_pow3 * x1_pow2 * x2_pow1 * factors[18]) % modulo +

			(x0_pow0 * x1_pow2 * x2_pow3 * factors[19]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow3 * factors[20]) % modulo +
			(x0_pow0 * x1_pow3 * x2_pow2 * factors[21]) % modulo +
			(x0_pow2 * x1_pow3 * x2_pow0 * factors[22]) % modulo +
			(x0_pow3 * x1_pow0 * x2_pow2 * factors[23]) % modulo +
			(x0_pow3 * x1_pow2 * x2_pow0 * factors[24]) % modulo +

			(x0_pow1 * x1_pow1 * x2_pow3 * factors[25]) % modulo +
			(x0_pow1 * x1_pow3 * x2_pow1 * factors[26]) % modulo +
			(x0_pow3 * x1_pow1 * x2_pow1 * factors[27]) % modulo +

			(x0_pow0 * x1_pow1 * x2_pow3 * factors[28]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow3 * factors[29]) % modulo +
			(x0_pow0 * x1_pow3 * x2_pow1 * factors[30]) % modulo +
			(x0_pow1 * x1_pow3 * x2_pow0 * factors[31]) % modulo +
			(x0_pow3 * x1_pow0 * x2_pow1 * factors[32]) % modulo +
			(x0_pow3 * x1_pow1 * x2_pow0 * factors[33]) % modulo +

			(x0_pow0 * x1_pow0 * x2_pow3 * factors[34]) % modulo +
			(x0_pow0 * x1_pow3 * x2_pow0 * factors[35]) % modulo +
			(x0_pow3 * x1_pow0 * x2_pow0 * factors[36]) % modulo +


			(x0_pow2 * x1_pow2 * x2_pow2 * factors[37]) % modulo +

			(x2_pow1 * x1_pow2 * x2_pow2 * factors[38]) % modulo +
			(x0_pow2 * x2_pow1 * x2_pow2 * factors[39]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow1 * factors[40]) % modulo +

			(x0_pow0 * x1_pow2 * x2_pow2 * factors[41]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow2 * factors[42]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow0 * factors[43]) % modulo +
			
			(x0_pow1 * x1_pow1 * x2_pow2 * factors[44]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow1 * factors[45]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow1 * factors[46]) % modulo +

			(x0_pow0 * x1_pow1 * x2_pow2 * factors[47]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow2 * factors[48]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow1 * factors[49]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow0 * factors[50]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow1 * factors[51]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow0 * factors[52]) % modulo +

			(x0_pow0 * x1_pow0 * x2_pow2 * factors[53]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow0 * factors[54]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow0 * factors[55]) % modulo +


			(x0_pow1 * x1_pow1 * x2_pow1 * factors[56]) % modulo +
			
			(x0_pow0 * x1_pow1 * x2_pow1 * factors[57]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow1 * factors[58]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow0 * factors[59]) % modulo +

			(x0_pow0 * x1_pow0 * x2_pow1 * factors[60]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow0 * factors[61]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow0 * factors[62]) % modulo +


			(x0_pow0 * x1_pow0 * x2_pow0 * factors[63])
		) % modulo;

		x0_pow1 = x1_pow1;
		x1_pow1 = x2_pow1;
		x2_pow1 = next_x;

		const int32_t idx = x0_pow1 * modulo * modulo + x1_pow1 * modulo + x2_pow1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_3_cycle_full_poly(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	values.assign(values.size(), 0);
	cycle_checker.assign(cycle_checker.size(), 0);

	switch (factors.size()) {
		case 8: return do_fac_3_cycle_full_poly_pow_1(modulo, cycle_len, factors, values, cycle_checker);
		case 27: return do_fac_3_cycle_full_poly_pow_2(modulo, cycle_len, factors, values, cycle_checker);
		case 64: return do_fac_3_cycle_full_poly_pow_3(modulo, cycle_len, factors, values, cycle_checker);
		default:
			assert(false);
			break;
	}
}

inline int32_t do_fac_4_cycle_full_poly_pow_1(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	const int32_t x0_pow0 = 1;
	const int32_t x1_pow0 = 1;
	const int32_t x2_pow0 = 1;
	const int32_t x3_pow0 = 1;

	int32_t x0_pow1 = 0;
	int32_t x1_pow1 = 0;
	int32_t x2_pow1 = 0;
	int32_t x3_pow1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*4 + 0] = x0_pow1;
		values[i*4 + 1] = x1_pow1;
		values[i*4 + 2] = x2_pow1;
		values[i*4 + 3] = x3_pow1;

		const int32_t next_x = (
			// 1 1 1 1
			(x0_pow1 * x1_pow1 * x2_pow1 * x3_pow1 * factors[0]) % modulo +
			// 1 1 1 0
			(x0_pow0 * x1_pow1 * x2_pow1 * x3_pow1 * factors[1]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow1 * x3_pow1 * factors[2]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow0 * x3_pow1 * factors[3]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow1 * x3_pow0 * factors[4]) % modulo +
			// 1 1 0 0
			(x0_pow0 * x1_pow0 * x2_pow1 * x3_pow1 * factors[5]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow0 * x3_pow1 * factors[6]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow1 * x3_pow0 * factors[7]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow0 * x3_pow1 * factors[8]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow1 * x3_pow0 * factors[9]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow0 * x3_pow0 * factors[10]) % modulo +
			// 1 0 0 0
			(x0_pow0 * x1_pow0 * x2_pow0 * x3_pow1 * factors[11]) % modulo +
			(x0_pow0 * x1_pow0 * x2_pow1 * x3_pow0 * factors[12]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow0 * x3_pow0 * factors[13]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow0 * x3_pow0 * factors[14]) % modulo +

			// 0 0 0 0
			(x0_pow0 * x1_pow0 * x2_pow0 * x3_pow0 * factors[15])
		) % modulo;

		x0_pow1 = x1_pow1;
		x1_pow1 = x2_pow1;
		x2_pow1 = x3_pow1;
		x3_pow1 = next_x;

		const int32_t idx = x0_pow1 * modulo * modulo * modulo + x1_pow1 * modulo * modulo + x2_pow1 * modulo + x3_pow1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_4_cycle_full_poly_pow_2(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	const int32_t x0_pow0 = 1;
	const int32_t x1_pow0 = 1;
	const int32_t x2_pow0 = 1;
	const int32_t x3_pow0 = 1;

	int32_t x0_pow1 = 0;
	int32_t x1_pow1 = 0;
	int32_t x2_pow1 = 0;
	int32_t x3_pow1 = 0;
	bool is_cycle_full = true;
	for (int32_t i = 0; i < cycle_len; ++i) {
		values[i*4 + 0] = x0_pow1;
		values[i*4 + 1] = x1_pow1;
		values[i*4 + 2] = x2_pow1;
		values[i*4 + 3] = x3_pow1;

		const int32_t x0_pow2 = (x0_pow1 * x0_pow1) % modulo;
		const int32_t x1_pow2 = (x1_pow1 * x1_pow1) % modulo;
		const int32_t x2_pow2 = (x2_pow1 * x2_pow1) % modulo;
		const int32_t x3_pow2 = (x3_pow1 * x3_pow1) % modulo;

		const int32_t next_x = (
			// 2 2 2 2
			(x0_pow2 * x1_pow2 * x2_pow2 * x3_pow2 * factors[0]) % modulo +
			// 2 2 2 1
			(x0_pow1 * x1_pow2 * x2_pow2 * x3_pow2 * factors[1]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow2 * x3_pow2 * factors[2]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow1 * x3_pow2 * factors[3]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow2 * x3_pow1 * factors[4]) % modulo +
			// 2 2 2 0
			(x0_pow0 * x1_pow2 * x2_pow2 * x3_pow2 * factors[5]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow2 * x3_pow2 * factors[6]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow0 * x3_pow2 * factors[7]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow2 * x3_pow0 * factors[8]) % modulo +
			// 2 2 1 1
			(x0_pow1 * x1_pow1 * x2_pow2 * x3_pow2 * factors[9]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow1 * x3_pow2 * factors[10]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow2 * x3_pow1 * factors[11]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow1 * x3_pow2 * factors[12]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow2 * x3_pow1 * factors[13]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow1 * x3_pow1 * factors[14]) % modulo +
			// 2 2 1 0
			(x0_pow0 * x1_pow1 * x2_pow2 * x3_pow2 * factors[15]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow2 * x3_pow2 * factors[16]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow1 * x3_pow2 * factors[17]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow0 * x3_pow2 * factors[18]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow2 * x3_pow1 * factors[19]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow2 * x3_pow0 * factors[20]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow1 * x3_pow2 * factors[21]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow0 * x3_pow2 * factors[22]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow2 * x3_pow1 * factors[23]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow2 * x3_pow0 * factors[24]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow0 * x3_pow1 * factors[25]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow1 * x3_pow0 * factors[26]) % modulo +
			// 2 2 0 0
			(x0_pow0 * x1_pow0 * x2_pow2 * x3_pow2 * factors[27]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow0 * x3_pow2 * factors[28]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow2 * x3_pow0 * factors[29]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow0 * x3_pow2 * factors[30]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow2 * x3_pow0 * factors[31]) % modulo +
			(x0_pow2 * x1_pow2 * x2_pow0 * x3_pow0 * factors[32]) % modulo +
			// 2 1 1 1
			(x0_pow1 * x1_pow1 * x2_pow1 * x3_pow2 * factors[33]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow2 * x3_pow1 * factors[34]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow1 * x3_pow1 * factors[35]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow1 * x3_pow1 * factors[36]) % modulo +
			// 2 1 1 0
			(x0_pow0 * x1_pow1 * x2_pow1 * x3_pow2 * factors[37]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow1 * x3_pow2 * factors[38]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow0 * x3_pow2 * factors[39]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow2 * x3_pow1 * factors[40]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow2 * x3_pow1 * factors[41]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow2 * x3_pow0 * factors[42]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow1 * x3_pow1 * factors[43]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow0 * x3_pow1 * factors[44]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow1 * x3_pow0 * factors[45]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow1 * x3_pow1 * factors[46]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow0 * x3_pow1 * factors[47]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow1 * x3_pow0 * factors[48]) % modulo +
			// 2 1 0 0
			(x0_pow0 * x1_pow0 * x2_pow1 * x3_pow2 * factors[49]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow0 * x3_pow2 * factors[50]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow0 * x3_pow2 * factors[51]) % modulo +
			(x0_pow0 * x1_pow0 * x2_pow2 * x3_pow1 * factors[52]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow2 * x3_pow0 * factors[53]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow2 * x3_pow0 * factors[54]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow0 * x3_pow1 * factors[55]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow1 * x3_pow0 * factors[56]) % modulo +
			(x0_pow1 * x1_pow2 * x2_pow0 * x3_pow0 * factors[57]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow0 * x3_pow1 * factors[58]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow1 * x3_pow0 * factors[59]) % modulo +
			(x0_pow2 * x1_pow1 * x2_pow0 * x3_pow0 * factors[60]) % modulo +
			// 2 0 0 0
			(x0_pow0 * x1_pow0 * x2_pow0 * x3_pow2 * factors[61]) % modulo +
			(x0_pow0 * x1_pow0 * x2_pow2 * x3_pow0 * factors[62]) % modulo +
			(x0_pow0 * x1_pow2 * x2_pow0 * x3_pow0 * factors[63]) % modulo +
			(x0_pow2 * x1_pow0 * x2_pow0 * x3_pow0 * factors[64]) % modulo +
			
			// 1 1 1 1
			(x0_pow1 * x1_pow1 * x2_pow1 * x3_pow1 * factors[65]) % modulo +
			// 1 1 1 0
			(x0_pow0 * x1_pow1 * x2_pow1 * x3_pow1 * factors[66]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow1 * x3_pow1 * factors[67]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow0 * x3_pow1 * factors[68]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow1 * x3_pow0 * factors[69]) % modulo +
			// 1 1 0 0
			(x0_pow0 * x1_pow0 * x2_pow1 * x3_pow1 * factors[70]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow0 * x3_pow1 * factors[71]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow1 * x3_pow0 * factors[72]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow0 * x3_pow1 * factors[73]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow1 * x3_pow0 * factors[74]) % modulo +
			(x0_pow1 * x1_pow1 * x2_pow0 * x3_pow0 * factors[75]) % modulo +
			// 1 0 0 0
			(x0_pow0 * x1_pow0 * x2_pow0 * x3_pow1 * factors[76]) % modulo +
			(x0_pow0 * x1_pow0 * x2_pow1 * x3_pow0 * factors[77]) % modulo +
			(x0_pow0 * x1_pow1 * x2_pow0 * x3_pow0 * factors[78]) % modulo +
			(x0_pow1 * x1_pow0 * x2_pow0 * x3_pow0 * factors[79]) % modulo +

			// 0 0 0 0
			(x0_pow0 * x1_pow0 * x2_pow0 * x3_pow0 * factors[80])
		) % modulo;

		x0_pow1 = x1_pow1;
		x1_pow1 = x2_pow1;
		x2_pow1 = x3_pow1;
		x3_pow1 = next_x;

		const int32_t idx = x0_pow1 * modulo * modulo * modulo + x1_pow1 * modulo * modulo + x2_pow1 * modulo + x3_pow1;
		if (cycle_checker[idx] == 1) {
			return 0;
		}

		cycle_checker[idx] = 1;
	}

	return 1;
}

inline int32_t do_fac_4_cycle_full_poly(
	const int32_t modulo,
	const int32_t cycle_len,
	const vector<int32_t>& factors,
	vector<int32_t>& values,
	vector<int32_t>& cycle_checker
) {
	values.assign(values.size(), 0);
	cycle_checker.assign(cycle_checker.size(), 0);

	switch (factors.size()) {
		case 16: return do_fac_4_cycle_full_poly_pow_1(modulo, cycle_len, factors, values, cycle_checker);
		case 81: return do_fac_4_cycle_full_poly_pow_2(modulo, cycle_len, factors, values, cycle_checker);
		default:
			assert(false);
			break;
	}
}

inline void do_fac_n_cycle_many_random(
	const RecordCycleManyRandomAmountNonzero& record_cycle_many_random_amount_nonzero,
	vector<RecordCycleFactors>& all_record_cycle_factors
) {
	const vector<uint64_t>& values_a = record_cycle_many_random_amount_nonzero.values_a;
	const vector<uint64_t>& values_c = record_cycle_many_random_amount_nonzero.values_c;
	const int32_t tries_per_thread = record_cycle_many_random_amount_nonzero.tries_per_thread;
	const int32_t factor_amount = record_cycle_many_random_amount_nonzero.factor_amount;
	const int32_t modulo = record_cycle_many_random_amount_nonzero.modulo;
	const int32_t factor_len = record_cycle_many_random_amount_nonzero.factor_len;
	const int32_t amount_nonzero_factors = record_cycle_many_random_amount_nonzero.amount_nonzero_factors;

	SimpleRandomNumberGenerator sRNG = SimpleRandomNumberGenerator(values_a, values_c);

	assert((factor_amount >= 1) && (factor_amount <= 4));
	const int32_t cycle_len = (
		factor_amount == 1 ? 
		modulo :
		factor_amount == 2 ?
		modulo * modulo :
		factor_amount == 3 ?
		modulo * modulo * modulo :
		modulo * modulo * modulo * modulo
	);
	cout << "do_fac_n_cycle_many_random: cycle_len: " << cycle_len << endl;
	vector<int32_t> t_factor_part;
	vector<int32_t> t_factor;
	vector<int32_t> t_cycle;
	vector<int32_t> cycle_checker;

	vector<int32_t> perm_pos_1;
	vector<int32_t> perm_pos_0;

	vector<int32_t> perm_pos_1_pos;
	vector<int32_t> perm_pos_0_pos;
	vector<int32_t> perm_pos_should_change;

	t_factor_part.resize(amount_nonzero_factors);
	t_factor.resize(factor_len);

	perm_pos_1.resize(amount_nonzero_factors - 1);
	perm_pos_0.resize(factor_len - amount_nonzero_factors);

	perm_pos_1_pos.resize(factor_len * factor_amount);
	perm_pos_0_pos.resize(factor_len * factor_amount);
	perm_pos_should_change.resize(factor_len * factor_len);

	t_cycle.resize(cycle_len * factor_amount);
	cycle_checker.resize(cycle_len);

	// call here in a loop do_fac_2_cycle_full_poly
	for (size_t i_tries = 0; i_tries < tries_per_thread; ++i_tries) {
		t_factor.assign(t_factor.size(), 0);

		// define the positions of the t_factor_part
		sRNG.calcNextRandomValModulo(amount_nonzero_factors, modulo - 1, t_factor_part);
		// sRNG.calcNextRandomValModulo(amount_nonzero_factors, 4, t_factor_part);

		// TODO: add a fixed t_cycle at the beginning for faster search!
		if (amount_nonzero_factors < factor_len) {
			for (size_t i = 0; i < amount_nonzero_factors - 1; ++i) {
				perm_pos_1[i] = i;
			}
			for (size_t i = 0; i < factor_len - amount_nonzero_factors; ++i) {
				perm_pos_0[i] = amount_nonzero_factors - 1 + i;
			}

			sRNG.calcNextRandomValModulo(perm_pos_1_pos.size(), perm_pos_1.size(), perm_pos_1_pos);
			sRNG.calcNextRandomValModulo(perm_pos_0_pos.size(), perm_pos_0.size(), perm_pos_0_pos);
			sRNG.calcNextRandomValModulo(perm_pos_should_change.size(), 2, perm_pos_should_change);

			const size_t size = perm_pos_1_pos.size();
			for (size_t i = 0; i < size; ++i) {
				if (perm_pos_should_change[i] == 1) {
					const int32_t pos_1 = perm_pos_1_pos[i];
					const int32_t pos_0 = perm_pos_0_pos[i];

					const int32_t pos_temp = perm_pos_1[pos_1];
					perm_pos_1[pos_1] = perm_pos_0[pos_0];
					perm_pos_0[pos_0] = pos_temp;
				}
			}

			for (size_t i = 0; i < amount_nonzero_factors - 1; ++i) {
				t_factor[perm_pos_1[i]] = t_factor_part[i] + 1;
			}
			t_factor[factor_len - 1] = t_factor_part[amount_nonzero_factors - 1] + 1;
		} else {
			for (size_t i = 0; i < amount_nonzero_factors; ++i) {
				t_factor[i] = t_factor_part[i] + 1;
			}
		}

		const int32_t ret_val = (
			factor_amount == 1 ?
			do_fac_1_cycle_full_poly(modulo, cycle_len, t_factor, t_cycle, cycle_checker) :
			factor_amount == 2 ?
			do_fac_2_cycle_full_poly(modulo, cycle_len, t_factor, t_cycle, cycle_checker) :
			factor_amount == 3 ?
			do_fac_3_cycle_full_poly(modulo, cycle_len, t_factor, t_cycle, cycle_checker) :
			do_fac_4_cycle_full_poly(modulo, cycle_len, t_factor, t_cycle, cycle_checker)
		);
		// const int32_t ret_val = do_fac_2_cycle_half_poly_one_fac_pow_1_max(modulo, cycle_len, t_factor, t_cycle, cycle_checker);

		if (ret_val == 1) {
			const chrono::time_point<chrono::high_resolution_clock> now = chrono::high_resolution_clock::now();
			std::time_t now2 = chrono::system_clock::to_time_t(now);
			std::string dt_str(30, '\0');
			std::strftime(&dt_str[0], dt_str.size(), "%Y-%m-%dT%H:%M:%S.", std::localtime(&now2));

			using days = chrono::duration<int, std::ratio_multiply<chrono::hours::period, std::ratio<24> >::type>;
			chrono::high_resolution_clock::duration tp = now.time_since_epoch();
			const days d = chrono::duration_cast<days>(tp);
			tp -= d;
			const chrono::hours h = chrono::duration_cast<chrono::hours>(tp);
			tp -= h;
			const chrono::minutes m = chrono::duration_cast<chrono::minutes>(tp);
			tp -= m;
			const chrono::seconds s = chrono::duration_cast<chrono::seconds>(tp);
			tp -= s;
			const chrono::milliseconds ms = chrono::duration_cast<chrono::milliseconds>(tp);
			tp -= ms;
			const chrono::microseconds us = chrono::duration_cast<chrono::microseconds>(tp);
			tp -= us;

			stringstream ss;
			dt_str.erase(std::find(dt_str.begin(), dt_str.end(), '\0'), dt_str.end());
			ss << dt_str
			 	<< std::setw(3) << std::setfill('0') << std::right << ms.count()
			 	<< std::setw(3) << std::setfill('0') << std::right << us.count();

			all_record_cycle_factors.push_back(RecordCycleFactors(
				factor_amount,
				modulo,
				cycle_len,
				factor_len,
				t_cycle,
				t_factor,
				amount_nonzero_factors,
				"new",
				ss.str()
			));
		}
	}
}

void do_fac_n_cycle_many_random_all_amount_nonzero(
	const RecordCycleManyRandom* record_cycle_many_random,
	vector<RecordCycleFactors>& all_record_cycle_factors_dt_sorted_filtered
) {
	vector<RecordCycleFactors> all_record_cycle_factors;
	const int32_t factor_len = record_cycle_many_random->factor_len;

	// for (int32_t amount_nonzero_i = 2; amount_nonzero_i < 8; ++amount_nonzero_i) {
	for (int32_t amount_nonzero_i = 2; amount_nonzero_i < factor_len + 1; ++amount_nonzero_i) {
		RecordCycleManyRandomAmountNonzero record_cycle_many_random_amount_nonzero = RecordCycleManyRandomAmountNonzero(
			*record_cycle_many_random,
			amount_nonzero_i
		);
		record_cycle_many_random_amount_nonzero.values_a.push_back(1 + 4 * amount_nonzero_i);
		record_cycle_many_random_amount_nonzero.values_c.push_back(1 + 2 * amount_nonzero_i);

		cout << "factor_amount: " << record_cycle_many_random->factor_amount << ", modulo: " << record_cycle_many_random->modulo << ", amount_nonzero_i: " << amount_nonzero_i << endl;
		// this_thread::sleep_for(chrono::milliseconds(5000));

		const size_t current_found_cycles = all_record_cycle_factors.size();
		do_fac_n_cycle_many_random(record_cycle_many_random_amount_nonzero, all_record_cycle_factors);
		const size_t next_found_cycles = all_record_cycle_factors.size();
		const size_t new_found_cycles = next_found_cycles - current_found_cycles;
		cout << "modulo: " << record_cycle_many_random->modulo << ", amount_nonzero_i: " << amount_nonzero_i << ", current found cycles: " << current_found_cycles << ", new_found_cycles: " << new_found_cycles << endl;
	}
	cout << "factor_amount: " << record_cycle_many_random->factor_amount << ", all_record_cycle_factors.size(): " << all_record_cycle_factors.size() << endl;

	vector<size_t> argsort_idx(all_record_cycle_factors.size());
	iota(argsort_idx.begin(), argsort_idx.end(), 0);
	stable_sort(argsort_idx.begin(), argsort_idx.end(),
		[&all_record_cycle_factors](size_t i1, size_t i2) {
			return (
				(all_record_cycle_factors[i1].t_cycle < all_record_cycle_factors[i2].t_cycle) ||
				(
					(all_record_cycle_factors[i1].amount_nonzero_factors < all_record_cycle_factors[i2].amount_nonzero_factors)  | (
						(all_record_cycle_factors[i1].amount_nonzero_factors == all_record_cycle_factors[i2].amount_nonzero_factors) &&
						(all_record_cycle_factors[i1].t_factor < all_record_cycle_factors[i2].t_factor)
					)
				)
			);
		}
	);

	if (argsort_idx.size() > 0) {
		RecordCycleFactors* prev_factors_values_dt = &(all_record_cycle_factors[argsort_idx[0]]);
		all_record_cycle_factors_dt_sorted_filtered.push_back(*prev_factors_values_dt);
		for (size_t i = 1; i < argsort_idx.size(); ++i) {
			const size_t idx = argsort_idx[i];
			RecordCycleFactors* next_factors_values_dt = &(all_record_cycle_factors[idx]);

			if (prev_factors_values_dt->t_cycle != next_factors_values_dt->t_cycle) {
				all_record_cycle_factors_dt_sorted_filtered.push_back(*next_factors_values_dt);
				prev_factors_values_dt = next_factors_values_dt;
			}
		}
	}

	cout << "all_record_cycle_factors_dt_sorted_filtered.size(): " << all_record_cycle_factors_dt_sorted_filtered.size() << endl;
}


// TODO: Use this until a test framework is used...
void simpleConsistenceTest(const bool print_values_cout) {
	// Test 1: calcNextRandomValNew
	{
		const vector<uint64_t> values_a = {0x123456789ABCDEF1UL+4UL*1, 0x89ABCDEF01234569UL};
		const vector<uint64_t> values_c = {0x2352452235FBBF43UL, 0x52235F23524BBF43UL};

		SimpleRandomNumberGenerator sRNG = SimpleRandomNumberGenerator(values_a, values_c);

		vector<uint64_t> vals;

		sRNG.calcNextRandomValNew(10, vals);
		// cout << "vals 1: " << vals << endl;
		assert(vals == vector({
			0xf96a9b330b87639cUL, 0x52f7b00dedf1fc77UL, 0x7fbccfcae1f83266UL, 0xce99d38fdf828b9cUL, 0x45faebb19be32102UL,
			0xd74ec629d6062ed7UL, 0x89e52486b8512eb0UL, 0x0c5ef00d9a20a070UL, 0x52a2b7dc3fed95a4UL, 0x38016226fb648cefUL
		}));

		sRNG.calcNextRandomValNew(10, vals);
		// cout << "vals 2: " << vals << endl;
		assert(vals == vector({
			0x85c8b768e2cbe766UL, 0x7650c9c82a7a3e3cUL, 0x410ff02850849caaUL, 0xcdd752009d149c0fUL, 0x8081e5c7b5e14af0UL,
			0xf460522d733d4380UL, 0xec0119367f3aef8cUL, 0x079c945faa0c7a07UL, 0x9d454583c39e7e86UL, 0x7e4c36bf0e444b7cUL
		}));
	}

	// Test 2: calcNextRandomValAdd
	{
		const vector<uint64_t> values_a = {0x123456789ABCDEF1UL+4UL*2, 0x89ABCDEF01234569UL};
		const vector<uint64_t> values_c = {0x2352452235FBBF43UL, 0x52235F23524BBF43UL};

		SimpleRandomNumberGenerator sRNG = SimpleRandomNumberGenerator(values_a, values_c);

		vector<uint64_t> vals;

		sRNG.calcNextRandomValAdd(10, vals);
		if (print_values_cout) {
			cout << "vals 1: " << vals << endl;
		}
		assert(vals == vector({
			0xdd18608676a8ecfcUL, 0x9c855acdcb59b23fUL, 0x219d9f31fa5ba1b6UL, 0x95c3058945e20294UL, 0x78e582a5e43c92aaUL,
			0xe574861ee226911fUL, 0xc1cc24abc2ceee00UL, 0xe10dfe3bf68ed7f8UL, 0x7ccf0b83ff22a554UL, 0x337904d93d5f1a27UL
		}));

		sRNG.calcNextRandomValAdd(5, vals);
		if (print_values_cout) {
			cout << "vals 2: " << vals << endl;
		}
		assert(vals == vector({
			0xdd18608676a8ecfcUL, 0x9c855acdcb59b23fUL, 0x219d9f31fa5ba1b6UL, 0x95c3058945e20294UL, 0x78e582a5e43c92aaUL,
			0xe574861ee226911fUL, 0xc1cc24abc2ceee00UL, 0xe10dfe3bf68ed7f8UL, 0x7ccf0b83ff22a554UL, 0x337904d93d5f1a27UL,
			0xa19e3eda41e6afa6UL, 0x524547cc4ce63a34UL, 0x807cb2ac05a0b0f2UL, 0xc3879041e1f84f47UL, 0xec822424ae09e5a0UL
		}));
	}

	// Test 3: calcNextRandomValNew and calcNextRandomValAdd
	{
		const vector<uint64_t> values_a = {0x123456789ABCDEF1UL+4UL*3, 0x89ABCDEF01234569UL};
		const vector<uint64_t> values_c = {0x2352452235FBBF43UL, 0x52235F23524BBF43UL};

		SimpleRandomNumberGenerator sRNG1 = SimpleRandomNumberGenerator(values_a, values_c);
		SimpleRandomNumberGenerator sRNG2 = SimpleRandomNumberGenerator(values_a, values_c);

		vector<uint64_t> vals_new;
		vector<uint64_t> vals_add;

		sRNG1.calcNextRandomValNew(15, vals_new);
		if (print_values_cout) {
			cout << "vals_new: " << vals_new << endl;
		}

		sRNG2.calcNextRandomValAdd(10, vals_add);
		sRNG2.calcNextRandomValAdd(5, vals_add);

		assert(vals_new == vals_add);
		assert(vals_new == vector({
			0x718a5da0e09ceffcUL, 0x23ddfe606e7d78c7UL, 0x2aa00d185ac61aa6UL, 0x66297a497edb3f6cUL, 0xbd38b126ac0b46b2UL,
			0xb6a7bdc3bcdefe67UL, 0xabeec9af7505c5b0UL, 0x2a5b4f498a919280UL, 0x7968076bffd92624UL, 0x78ba3751a4a2641fUL,
			0xe0df22d713426286UL, 0x1630940ffb68ec8cUL, 0xdafe3ee1f6596afaUL, 0x89e4f5c569476d7fUL, 0x905152550acf1750UL
		}));
	}

	// Test 4: values_a and values_c minimum
	{
		const vector<uint64_t> values_a_1 = {0x0000000000000001UL};
		const vector<uint64_t> values_c_1 = {0x0000000000000001UL};
		const vector<uint64_t> values_a_2 = {0x0000000000000001UL, 0x0000000000000001UL};
		const vector<uint64_t> values_c_2 = {0x0000000000000001UL, 0x0000000000000001UL};
		const vector<uint64_t> values_a_3 = {0x0000000000000001UL, 0x0000000000000001UL, 0x0000000000000001UL};
		const vector<uint64_t> values_c_3 = {0x0000000000000001UL, 0x0000000000000001UL, 0x0000000000000001UL};

		SimpleRandomNumberGenerator sRNG_1 = SimpleRandomNumberGenerator(values_a_1, values_c_1);
		SimpleRandomNumberGenerator sRNG_2 = SimpleRandomNumberGenerator(values_a_2, values_c_2);
		SimpleRandomNumberGenerator sRNG_3 = SimpleRandomNumberGenerator(values_a_3, values_c_3);

		vector<uint64_t> vals_new_1;
		vector<uint64_t> vals_new_2;
		vector<uint64_t> vals_new_3;

		sRNG_1.calcNextRandomValNew(10, vals_new_1);
		sRNG_2.calcNextRandomValNew(10, vals_new_2);
		sRNG_3.calcNextRandomValNew(10, vals_new_3);

		if (print_values_cout) {
			cout << "vals_new_1: " << vals_new_1 << endl;
			cout << "vals_new_2: " << vals_new_2 << endl;
			cout << "vals_new_3: " << vals_new_3 << endl;
		}

		assert(vals_new_1 == vector({
			0x7ee524b50e992c1aUL, 0x094c55b976e30eb4UL, 0x4455833384607441UL, 0x36115f8dc0e71dabUL, 0x821e422501b8c922UL,
			0xbdc69f3c897b86c0UL, 0x99241979bfb1a629UL, 0xcbc4211dc9929ab7UL, 0xfdd5caac506bd2baUL, 0x99ef3f6b44ec4aacUL
		}));
		assert(vals_new_2 == vector({
			0x77a9710c787a22afUL, 0x4d19d68af2837af4UL, 0x05edadb23cfd4b59UL, 0xf916cb2233dcae7bUL, 0x3a3570abb43e04b9UL,
			0xddf44d6705168e94UL, 0x68d548cfc21d3827UL, 0xebe5a6d69cefc697UL, 0x0cefbd08d69aa03fUL, 0x2919ee18d3b0f744UL
		}));
		assert(vals_new_3 == vector({
			0x4d19d68af2837af4UL, 0x7244dcbe448769e9UL, 0xb40f1da8c15fd48eUL, 0x72c10b937a403516UL, 0x56a65afb724d4902UL,
			0xe6ef25ccb77ce812UL, 0x44d0e022e3b97d15UL, 0x329caf3c66cad114UL, 0x24136d02f823d9cfUL, 0x4cbc02f3f9e2b747UL
		}));
	}

	// Test 5: calcNextRandomValNew and calcNextRandomValAdd
	{
		const vector<uint64_t> values_a_1 = {0x123456789ABCDEF1UL+4UL*3, 0x89ABCDEF01234569UL};
		const vector<uint64_t> values_c_1 = {0x2352452235FBBF43UL, 0x52235F23524BBF43UL};
		const vector<uint64_t> values_a_2 = {0x123456789ABCDEF1UL+4UL*3, 0x89ABCDEF01234569UL, 0x0000000000000001UL};
		const vector<uint64_t> values_c_2 = {0x2352452235FBBF43UL, 0x52235F23524BBF43UL, 0x0000000000000001UL};

		SimpleRandomNumberGenerator sRNG1 = SimpleRandomNumberGenerator(values_a_1, values_c_1);
		SimpleRandomNumberGenerator sRNG2 = SimpleRandomNumberGenerator(values_a_2, values_c_2);

		vector<uint64_t> vals_new_1;
		vector<uint64_t> vals_new_2;

		sRNG1.calcNextRandomValNew(10, vals_new_1);
		sRNG2.calcNextRandomValNew(10, vals_new_2);

		if (print_values_cout) {
			cout << "vals_new_1: " << vals_new_1 << endl;
			cout << "vals_new_2: " << vals_new_2 << endl;
		}

		assert(vals_new_1 == vector({
			0x718a5da0e09ceffcUL, 0x23ddfe606e7d78c7UL, 0x2aa00d185ac61aa6UL, 0x66297a497edb3f6cUL, 0xbd38b126ac0b46b2UL,
			0xb6a7bdc3bcdefe67UL, 0xabeec9af7505c5b0UL, 0x2a5b4f498a919280UL, 0x7968076bffd92624UL, 0x78ba3751a4a2641fUL
		}));
		assert(vals_new_2 == vector({
			0xbc939816b7778a80UL, 0xceab9da2952e1edeUL, 0xbe35b074ef1ebab8UL, 0x1b770ad06ff9d020UL, 0x0c0685f230281eb9UL,
			0xecd58810993d8624UL, 0xf15a6c52002366e1UL, 0x85e3d202dbd4ae33UL, 0x2e29c0ded662b7f9UL, 0xa8ab4174e93043f1UL
		}));
	}

	// Test 5: calcNextRandomValModulo
	{
		const vector<uint64_t> values_a = {0x123456789ABCDEF1UL+4UL*4, 0x89ABCDEF01234569UL};
		const vector<uint64_t> values_c = {0x2352452235FBBF43UL, 0x52235F23524BBF43UL};

		SimpleRandomNumberGenerator sRNG = SimpleRandomNumberGenerator(values_a, values_c);

		vector<int32_t> vals_modulo;

		sRNG.calcNextRandomValModulo(100, 7, vals_modulo);
		if (print_values_cout) {
			cout << "vals_modulo: " << vals_modulo << endl;
		}
		assert(vals_modulo == vector({
			0, 3, 5, 4, 6, 0, 3, 3, 5, 5, 5, 1, 0, 5, 1, 1, 0, 1, 5, 0, 6, 0, 3, 6, 2, 4, 4, 0, 1, 4,
			5, 4, 0, 6, 4, 2, 0, 1, 6, 2, 0, 2, 1, 6, 1, 0, 5, 0, 0, 0, 6, 6, 5, 5, 2, 6, 2, 6, 3, 0,
			1, 6, 2, 2, 1, 0, 4, 5, 5, 5, 5, 4, 3, 5, 1, 6, 3, 6, 3, 3, 3, 0, 6, 6, 2, 6, 6, 4, 0, 0,
			2, 6, 2, 0, 2, 5, 5, 4, 6, 1
		}));

		sRNG.calcNextRandomValModulo(100, 10, vals_modulo);
		if (print_values_cout) {
			cout << "vals_modulo: " << vals_modulo << endl;
		}
		assert(vals_modulo == vector({
			3, 5, 6, 5, 5, 3, 0, 4, 9, 4, 6, 8, 2, 3, 1, 2, 0, 5, 3, 1, 5, 5, 3, 7, 7, 6, 7, 3, 7, 5,
			5, 1, 3, 6, 4, 5, 4, 9, 9, 1, 3, 5, 3, 2, 1, 5, 3, 8, 6, 4, 2, 0, 8, 5, 6, 4, 6, 6, 0, 6,
			2, 5, 1, 3, 8, 4, 9, 6, 3, 7, 5, 1, 7, 9, 0, 5, 3, 6, 5, 1, 8, 6, 4, 1, 5, 1, 4, 0, 1, 4,
			5, 2, 2, 5, 8, 1, 3, 2, 1, 1
		}));
	}
}

// from: https://stackoverflow.com/questions/35287746/merge-vector-of-vectors-into-a-single-vector
template<typename T>
void joinVectorOfVectors(vector<vector<T>>& vector_of_vectors_of_T, vector<T>& vector_of_T) {
	size_t total_size = 0;
	for (auto const& items: vector_of_vectors_of_T){
		total_size += items.size();
	}

	vector_of_T.reserve(total_size);
	for (auto& items: vector_of_vectors_of_T){
		std::move(items.begin(), items.end(), std::back_inserter(vector_of_T));
	}
}

static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
	int i;
	for(i = 0; i<argc; i++) {
		printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
	}
	printf("\n");
	return 0;
}

void trim(string& s, const string& trim_of) {
	size_t start = s.find_first_not_of(trim_of);
	size_t end = s.find_last_not_of(trim_of);

	if (start == string::npos)
		s.clear();
	else
		s = s.substr(start, end - start + 1);
}

string trim(const string& s, const string& trim_of) {
	string s_cpy = s;
	size_t start = s_cpy.find_first_not_of(trim_of);
	size_t end = s_cpy.find_last_not_of(trim_of);

	if (start == string::npos) {
		s_cpy.clear();
	} else {
		s_cpy = s_cpy.substr(start, end - start + 1);
	}

	return s_cpy;
}

static int callbackSelectImportRecordCycleFactor(void *data, int argc, char **argv, char **azColName) {
	vector<RecordIdCycleFactors>* vec_record_id_cycle_factors = (vector<RecordIdCycleFactors>*)data;

	size_t pos;
	const int32_t id = std::stoi(argv[0], &pos, 10);
	const int32_t factor_amount = std::stoi(argv[1], &pos, 10);
	const int32_t modulo = std::stoi(argv[2], &pos, 10);
	const int32_t cycle_len = std::stoi(argv[3], &pos, 10);
	const int32_t factor_len = std::stoi(argv[4], &pos, 10);
	
	const vector<string> t_cycle_str = (
		factor_amount == 1 ?
		split(string(argv[5]), string("),(")) :
		split(string(argv[5]), string(","))
	);
	vector<int32_t> t_cycle;
	for (size_t i = 0; i < t_cycle_str.size(); ++i) {
		t_cycle.push_back(std::stoi(trim(t_cycle_str[i], "() ,").c_str(), &pos, 10));
	}

	const vector<string> t_factor_str = split(string(argv[6]), string(","));
	vector<int32_t> t_factor;
	for (size_t i = 0; i < t_factor_str.size(); ++i) {
		t_factor.push_back(std::stoi(trim(t_factor_str[i], "() ").c_str(), &pos, 10));
	}

	const int32_t amount_nonzero_factors = std::stoi(argv[7], &pos, 10);
	const string status = argv[8];
	const string dt = argv[9];

	vec_record_id_cycle_factors->push_back(RecordIdCycleFactors(
		id,
		factor_amount,
		modulo,
		cycle_len,
		factor_len,
		t_cycle,
		t_factor,
		amount_nonzero_factors,
		status,
		dt
	));

	return 0;
}

void extendInsertStringStreamSQL(stringstream& ss, const string& sql_query_insert_template, const RecordCycleFactors& record_cycle_factors) {
	ss << sql_query_insert_template;
	ss << "(";

	ss << record_cycle_factors.factor_amount;
	ss << "," << record_cycle_factors.modulo;
	ss << "," << record_cycle_factors.cycle_len;
	ss << "," << record_cycle_factors.factor_len << ",";
	// t_cycle
	const vector<int32_t>& t_cycle = record_cycle_factors.t_cycle;
	ss << "\"(";
	if (record_cycle_factors.factor_amount == 1) {
		ss << "(" << t_cycle[0] << ",)";
		for (size_t j = 1; j < record_cycle_factors.cycle_len*1; j += 1) {
			ss << ",(" << t_cycle[j] << ",)";
		}
	} else if (record_cycle_factors.factor_amount == 2) {
		ss << "(" << t_cycle[0] << "," << t_cycle[1] << ")";
		for (size_t j = 2; j < record_cycle_factors.cycle_len*2; j += 2) {
			ss << ",(" << t_cycle[j] << "," << t_cycle[j+1] << ")";
		}
	} else if (record_cycle_factors.factor_amount == 3) {
		ss << "(" << t_cycle[0] << "," << t_cycle[1] << "," << t_cycle[2] << ")";
		for (size_t j = 3; j < record_cycle_factors.cycle_len*3; j += 3) {
			ss << ",(" << t_cycle[j] << "," << t_cycle[j+1] << "," << t_cycle[j+2] << ")";
		}
	}  else if (record_cycle_factors.factor_amount == 4) {
		ss << "(" << t_cycle[0] << "," << t_cycle[1] << "," << t_cycle[2] << "," << t_cycle[3] << ")";
		for (size_t j = 4; j < record_cycle_factors.cycle_len*4; j += 4) {
			ss << ",(" << t_cycle[j] << "," << t_cycle[j+1] << "," << t_cycle[j+2] << "," << t_cycle[j+3] << ")";
		}
	} else {
		assert(false);
	}
	ss << ")\"";
	ss << ",";
	// t_factor
	const vector<int32_t>& t_factor = record_cycle_factors.t_factor;
	ss << "\"(";
	ss << t_factor[0];
	for (size_t j = 1; j < record_cycle_factors.factor_len; j += 1) {
		ss << "," << t_factor[j];
	}
	ss << ")\"";
	ss << ",";
	ss << record_cycle_factors.amount_nonzero_factors;
	ss << ",";
	ss << "\"" << record_cycle_factors.status << "\"";
	ss << ",";
	ss << "\"" << record_cycle_factors.dt << "\"";
	ss << ");";
	ss << endl;
}

void extendDeleteStringStreamSQL(stringstream& ss, const string& sql_query_delete_template, const RecordIdCycleFactors& record_id_cycle_factors) {
	ss << sql_query_delete_template;
	ss << record_id_cycle_factors.id << ";" << endl;
}

int main(int argc, char* argv[]) {
	// TODO: make function call tests only when in args tests=1 is written
	// simpleConsistenceTest(false);

	// A simple arg parser
	map<string, string> args;
	for (size_t i = 1; i < argc; ++i) {
		vector<string> arg_split = split(string(argv[i]), string("="));
		assert(arg_split.size() == 2);
		args[arg_split[0]] = arg_split[1];
	}

	vector<uint64_t> values_a;
	vector<uint64_t> values_c;
	int32_t tries_per_thread;
	int32_t factor_amount;
	int32_t modulo;
	int32_t factor_len;
	string file_path;
	string sqlite_file_path;
	int32_t thread_count_available;

	for (map<string, string>::iterator it = args.begin(); it != args.end(); ++it) {
		const string arg_name = it->first;
		const string arg_val = it->second;

		if (arg_name == "values_a") {
			for (string& val: split(arg_val, string(","))) {
				uint64_t num;
				if (sscanf(val.c_str(), "%" SCNx64, &num) != 1) {
					assert(false);
				} 
				values_a.push_back(num);
			}
		} else if (arg_name == "values_c") {
			for (string& val: split(arg_val, string(","))) {
				uint64_t num;
				if (sscanf(val.c_str(), "%" SCNx64, &num) != 1) {
					assert(false);
				} 
				values_c.push_back(num);
			}
		} else if (arg_name == "tries_per_thread") {
			size_t pos;
			tries_per_thread = std::stoi(arg_val.c_str(), &pos, 10);
		} else if (arg_name == "factor_amount") {
			size_t pos;
			factor_amount = std::stoi(arg_val.c_str(), &pos, 10);
		} else if (arg_name == "modulo") {
			size_t pos;
			modulo = std::stoi(arg_val.c_str(), &pos, 10);
		} else if (arg_name == "factor_len") {
			size_t pos;
			factor_len = std::stoi(arg_val.c_str(), &pos, 10);
		} else if (arg_name == "thread_count_available") {
			size_t pos;
			thread_count_available = std::stoi(arg_val.c_str(), &pos, 10);
		} else if (arg_name == "file_path") {
			file_path = arg_val;
		} else if (arg_name == "sqlite_file_path") {
			sqlite_file_path = arg_val;
		}
	}

	// TODO: make it multithreading able!
	cout << "args: " << args << endl;
	cout << "values_a: " << values_a << endl;
	cout << "values_c: " << values_c << endl;
	cout << "tries_per_thread: " << tries_per_thread << endl;
	cout << "factor_amount: " << factor_amount << endl;
	cout << "modulo: " << modulo << endl;
	cout << "factor_len: " << factor_len << endl;
	cout << "thread_count_available: " << thread_count_available << endl;
	cout << "file_path: " << file_path << endl;

	vector<thread> vec_thread(thread_count_available);
	vector<RecordCycleManyRandom*> vec_record_cycle_many_random;
	vector<vector<RecordCycleFactors>> vec_vec_record_cycle_factors(thread_count_available);

	for (size_t worker_id = 0; worker_id < thread_count_available; ++worker_id) {
		vec_record_cycle_many_random.push_back(new RecordCycleManyRandom(
			values_a,
			values_c,
			tries_per_thread,
			factor_amount,
			modulo,
			factor_len
		));
		RecordCycleManyRandom* record_cycle_many_random = vec_record_cycle_many_random[worker_id];
		record_cycle_many_random->values_a.push_back(1 + worker_id * 4);
		record_cycle_many_random->values_c.push_back(1 + worker_id * 4);

		vector<RecordCycleFactors>& all_record_cycle_factors_dt_sorted_filtered = vec_vec_record_cycle_factors[worker_id];

		vec_thread[worker_id] = thread(
			&do_fac_n_cycle_many_random_all_amount_nonzero,
			record_cycle_many_random,
			ref(all_record_cycle_factors_dt_sorted_filtered)
		);

		// do_fac_n_cycle_many_random_all_amount_nonzero(record_cycle_many_random, all_record_cycle_factors_dt_sorted_filtered);
	}

	for (size_t worker_id = 0; worker_id < thread_count_available; ++worker_id) {
		vec_thread[worker_id].join();
		delete vec_record_cycle_many_random[worker_id];
	}

	vector<RecordCycleFactors> vec_record_cycle_factors_dt_merged;
	joinVectorOfVectors(vec_vec_record_cycle_factors, vec_record_cycle_factors_dt_merged);
	vector<RecordCycleFactors> vec_record_cycle_factors_dt_merged_filtered;

	vector<size_t> argsort_idx(vec_record_cycle_factors_dt_merged.size());
	iota(argsort_idx.begin(), argsort_idx.end(), 0);
	stable_sort(argsort_idx.begin(), argsort_idx.end(),
		[&vec_record_cycle_factors_dt_merged](size_t i1, size_t i2) {
			return (
				(vec_record_cycle_factors_dt_merged[i1].t_cycle < vec_record_cycle_factors_dt_merged[i2].t_cycle) ||
				(
					(vec_record_cycle_factors_dt_merged[i1].t_cycle == vec_record_cycle_factors_dt_merged[i2].t_cycle) &&
					(vec_record_cycle_factors_dt_merged[i1].t_factor < vec_record_cycle_factors_dt_merged[i2].t_factor)
				)
			);
		}
	);

	if (argsort_idx.size() > 0) {
		RecordCycleFactors* prev_factors_values_dt = &(vec_record_cycle_factors_dt_merged[argsort_idx[0]]);
		vec_record_cycle_factors_dt_merged_filtered.push_back(*prev_factors_values_dt);
		for (size_t i = 1; i < argsort_idx.size(); ++i) {
			const size_t idx = argsort_idx[i];
			RecordCycleFactors* next_factors_values_dt = &(vec_record_cycle_factors_dt_merged[idx]);

			if (prev_factors_values_dt->t_cycle != next_factors_values_dt->t_cycle) {
				vec_record_cycle_factors_dt_merged_filtered.push_back(*next_factors_values_dt);
				prev_factors_values_dt = next_factors_values_dt;
			}
		}
	}

	cout << "vec_record_cycle_factors_dt_merged_filtered.size(): " << vec_record_cycle_factors_dt_merged_filtered.size() << endl;

	// const RecordCycleManyRandom record_cycle_many_random = RecordCycleManyRandom(
	// 	values_a,
	// 	values_c,
	// 	tries_per_thread,
	// 	modulo,
	// 	factor_len
	// );
	// vector<RecordCycleFactors> all_record_cycle_factors_dt_sorted_filtered;

	// do_fac_n_cycle_many_random_all_amount_nonzero(record_cycle_many_random, all_record_cycle_factors_dt_sorted_filtered);

	// cout << "all_record_cycle_factors_dt_sorted_filtered.size(): " << all_record_cycle_factors_dt_sorted_filtered.size() << endl;

	// ofstream o_file(file_path);

	// o_file << "args: " << args << endl;
	// o_file << "modulo|cycle_len|factor_len|t_cycle|t_factor|amount_nonzero_factors|status|dt" << endl;

	// for (size_t i = 0; i < vec_record_cycle_factors_dt_merged_filtered.size(); ++i) {
	// 	const RecordCycleFactors& record_cycle_factors = vec_record_cycle_factors_dt_merged_filtered[i];

	// 	o_file << record_cycle_factors.modulo;
	// 	o_file << "|" << record_cycle_factors.cycle_len;
	// 	o_file << "|" << record_cycle_factors.factor_len << "|";
	// 	// t_cycle
	// 	const vector<int32_t>& t_cycle = record_cycle_factors.t_cycle;
	// 	o_file << "(";
	// 	o_file << "(" << t_cycle[0] << "," << t_cycle[1] << ")";
	// 	for (size_t j = 2; j < record_cycle_factors.cycle_len*2; j += 2) {
	// 		o_file << ",(" << t_cycle[j] << "," << t_cycle[j+1] << ")";
	// 	}
	// 	o_file << ")";
	// 	o_file << "|";
	// 	// t_factor
	// 	const vector<int32_t>& t_factor = record_cycle_factors.t_factor;
	// 	o_file << "(";
	// 	o_file << t_factor[0];
	// 	for (size_t j = 1; j < factor_len; j += 1) {
	// 		o_file << "," << t_factor[j];
	// 	}
	// 	o_file << ")";
	// 	o_file << "|";
	// 	o_file << record_cycle_factors.amount_nonzero_factors;
	// 	o_file << "|";
	// 	o_file << "new";
	// 	o_file << "|";
	// 	o_file << record_cycle_factors.dt;
	// 	o_file << endl;
	// }

	// o_file.close();


	sqlite3* DB;
	char *zErrMsg = 0;
	string sql;
	int rc = 0;
	
	rc = sqlite3_open(sqlite_file_path.c_str(), &DB);

	if (rc) {
		std::cerr << "Error open DB " << sqlite3_errmsg(DB) << std::endl;
		return (-1);
	}
	else {
		std::cout << "Opened Database Successfully!" << std::endl;
	}

	/* Create SQL statement */
	sql = 
// "CREATE TABLE IF NOT EXISTS cyclic_2_factor_sequence (\n"
"CREATE TABLE IF NOT EXISTS cyclic_n_factor_sequence (\n"
"	\"id\" INTEGER PRIMARY KEY AUTOINCREMENT,\n"
"   \"factor_amount\" INTEGER NOT NULL,\n"
"	\"modulo\" INTEGER NOT NULL,\n"
"	\"cycle_len\" INTEGER NOT NULL,\n"
"	\"factor_len\" INTEGER NOT NULL,\n"
"	\"t_cycle\" TEXT NOT NULL,\n"
"	\"t_factor\" TEXT NOT NULL,\n"
"	\"amount_nonzero_factors\" INTEGER NOT NULL,\n"
"	\"status\" TEXT NOT NULL,\n"
"	\"dt\" TEXT NOT NULL,\n"
"	UNIQUE(factor_amount, modulo, factor_len, t_cycle)\n"
");";

	/* Execute SQL statement */
	rc = sqlite3_exec(DB, sql.c_str(), callback, 0, &zErrMsg);
	
	if( rc != SQLITE_OK ){
		fprintf(stderr, "SQL error: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	} else {
		fprintf(stdout, "Table created successfully\n");
	}

	const string query_select_tbl_cyclic_2_factor_sequence_complete =
"SELECT\n"
"	id, factor_amount, modulo, cycle_len, factor_len, t_cycle, t_factor, amount_nonzero_factors, status, dt\n"
"FROM\n"
"	cyclic_n_factor_sequence\n"
;
	stringstream query_select;
	query_select << query_select_tbl_cyclic_2_factor_sequence_complete;
	query_select << "WHERE\n";
	query_select << "	factor_amount = " << factor_amount << " AND modulo = " << modulo << " AND factor_len = " << factor_len;
	query_select << ";\n";

	vector<RecordIdCycleFactors> vec_record_id_cycle_factors;
	rc = sqlite3_exec(DB, query_select.str().c_str(), callbackSelectImportRecordCycleFactor, (void*)&vec_record_id_cycle_factors, &zErrMsg);
	
	if( rc != SQLITE_OK ){
		fprintf(stderr, "SQL error: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	} else {
		fprintf(stdout, "Records read successfully\n");
	}

	// TODO: find already created DB entries of t_cycle and t_factor
	// TODO: find the smallest t_factor, if entry exsits and delete the already entry and insert the new one with status update
	// TODO: otherwise if the new entry does not exist, insert it to the db 

	// set<RecordCycleFactors*> set_record_id_cycle_factor_from_db;
	pointer_set<RecordCycleFactors> set_record_id_cycle_factor_from_db;
	for (size_t i = 0; i < vec_record_id_cycle_factors.size(); ++i) {
		set_record_id_cycle_factor_from_db.insert(&(vec_record_id_cycle_factors[i]));
		// set_record_id_cycle_factor_from_db.insert(move(vec_record_id_cycle_factors[i]));
	}

	const string sql_query_insert_template =
"INSERT INTO \"cyclic_n_factor_sequence\" (\"factor_amount\", \"modulo\", \"cycle_len\", \"factor_len\", \"t_cycle\", \"t_factor\", \"amount_nonzero_factors\", \"status\", \"dt\") VALUES\n"
// "	({{modulo}}, {{cycle_len}}, {{factor_len}}, \"{{t_cycle}}\", \"{{t_factor}}\",\n"
// "		{{amount_nonzero_factors}}, \"{{status}}\", \"{{dt}}\");\n"
;
	
	const string sql_query_delete_template =
"DELETE FROM \"cyclic_n_factor_sequence\"\n"
"WHERE id = "
;

	int32_t count_new_insert = 0;
	int32_t count_update_insert = 0;
	stringstream ss_query_insert_delete;
	for (size_t i = 0; i < vec_record_cycle_factors_dt_merged_filtered.size(); ++i) {
		// RecordCycleFactors& record_cycle_factors = vec_record_cycle_factors_dt_merged_filtered[i];
		RecordCycleFactors* record_cycle_factors = &(vec_record_cycle_factors_dt_merged_filtered[i]);

		// if (set_record_id_cycle_factor_from_db.contains(record_cycle_factors)) {
		if (set_record_id_cycle_factor_from_db.contains(record_cycle_factors)) {
			// RecordIdCycleFactors& record_id_cycle_factors = *(RecordIdCycleFactors*)&(*set_record_id_cycle_factor_from_db.find(record_cycle_factors));
			RecordIdCycleFactors& record_id_cycle_factors = *static_cast<RecordIdCycleFactors* const>(*set_record_id_cycle_factor_from_db.find(record_cycle_factors));
			if (
				(record_cycle_factors->amount_nonzero_factors < record_id_cycle_factors.amount_nonzero_factors) ||
				(
					(record_cycle_factors->amount_nonzero_factors == record_id_cycle_factors.amount_nonzero_factors) &&
					(record_cycle_factors->t_factor < record_id_cycle_factors.t_factor)
				)
			) {
				cout << "found modulo, factor_len, t_cycle same:" << endl;
				cout << "record_cycle_factors: " << *record_cycle_factors << endl;
				cout << "record_id_cycle_factors: " << record_id_cycle_factors << endl;
				extendDeleteStringStreamSQL(ss_query_insert_delete, sql_query_delete_template, record_id_cycle_factors);

				// record_cycle_factors.status = "update";
				// extendInsertStringStreamSQL(ss_query_insert_delete, sql_query_insert_template, record_cycle_factors);
				record_cycle_factors->status = "update";
				extendInsertStringStreamSQL(ss_query_insert_delete, sql_query_insert_template, *record_cycle_factors);
				count_update_insert += 1;
			}
		} else {
			extendInsertStringStreamSQL(ss_query_insert_delete, sql_query_insert_template, *record_cycle_factors);
			count_new_insert += 1;
		}
	}

	cout << "count_new_insert: " << count_new_insert << endl;
	cout << "count_update_insert: " << count_update_insert << endl;

	rc = sqlite3_exec(DB, ss_query_insert_delete.str().c_str(), callback, 0, &zErrMsg);
	
	if( rc != SQLITE_OK ){
		fprintf(stderr, "SQL error: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	} else {
		fprintf(stdout, "Records created successfully\n");
	}

	// TODO: print the current state of how many modulo, factor_len, t_cycle amount are in the DB

	sqlite3_close(DB);
	exit(0);

	stringstream ss_query_insert;
	for (const RecordCycleFactors& record_cycle_factors: vec_record_cycle_factors_dt_merged_filtered) {
		extendInsertStringStreamSQL(ss_query_insert, sql_query_insert_template, record_cycle_factors);
	}

	/* Execute SQL statement */
	const string string_query_insert = ss_query_insert.str();
	// cout << "string_query_insert:" << endl;
	// cout << string_query_insert;

	rc = sqlite3_exec(DB, string_query_insert.c_str(), callback, 0, &zErrMsg);
	
	if( rc != SQLITE_OK ){
		fprintf(stderr, "SQL error: %s\n", zErrMsg);
		sqlite3_free(zErrMsg);
	} else {
		fprintf(stdout, "Records created successfully\n");
	}

	sqlite3_close(DB);

	return 0;
}
