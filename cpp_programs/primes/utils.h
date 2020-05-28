#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <ostream>
#include <vector>
#include <string>
#include <stdint.h>

using std::vector;
using std::ostream;
using std::for_each;
using std::string;

ostream& operator<<(ostream& os, const std::vector<uint8_t>& obj);

ostream& operator<<(ostream& os, const std::vector<string>& obj);

template<typename T>
ostream& operator<<(ostream& os, const std::vector<T>& obj);

#endif // UTILS_H
