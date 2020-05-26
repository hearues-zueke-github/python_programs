#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <ostream>
#include <vector>
#include <stdint.h>

using std::vector;
using std::ostream;
using std::for_each;

ostream& operator<<(ostream& os, const std::vector<uint8_t>& obj);

template<typename T>
ostream& operator<<(ostream& os, const std::vector<T>& obj);

#endif // UTILS_H
