#ifndef UTILS_PRIMES_H
#define UTILS_PRIMES_H

#include <vector>
#include <ostream>
#include <algorithm>
#include <stdint.h>

using std::vector;
using std::ostream;
using std::for_each;

template<typename T>
ostream& operator<<(ostream& os, const std::vector<T>& obj);

template<typename T>
void subVector(const vector<T>& src, vector<T>& dst, int m, int n);

inline uint64_t intSqrt(const uint64_t n) {
  uint64_t n_1 = (n + 1) / 2;
  uint64_t n_2 = (n_1 + n / n_1) / 2;

  while (true) {
    uint64_t n_3 = (n_2 + n / n_2) / 2;
    if (n_3 == n_1) {
      return n_1;
    }
    n_1 = n_2;
    n_2 = n_3;
  }
}

#endif // UTILS_PRIMES_H
