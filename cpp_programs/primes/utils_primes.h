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

#endif // UTILS_PRIMES_H
