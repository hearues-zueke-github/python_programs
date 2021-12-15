#ifndef UTILS_H
#define UTILS_H

#include <cstdint>

using U32 = uint32_t;
using U64 = uint64_t;

#include <vector>
#include <iomanip>
#include <ostream>
#include <sstream>

#include <algorithm>
#include <climits>
#include <limits>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <type_traits>
#include <unordered_set>
#include <map>

using namespace std;

template<typename T>
ostream& operator<<(ostream& os, const std::vector<T>& obj);

template<typename T>
ostream& operator<<(ostream& os, const unordered_set<T>& obj);

template<typename K, typename V>
ostream& operator<<(ostream& os, const std::map<K, V>& obj);

#endif // UTILS_H
