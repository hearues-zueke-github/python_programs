#ifndef UTILS_H
#define UTILS_H

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
// #include <tuple>

using namespace std;

template<typename T, size_t count, int byte>
struct byte_repeater;

template<typename T, int byte>
struct byte_repeater<T, 1, byte> {
    static constexpr T value = byte;
};

template<typename T, size_t count, int byte>
struct byte_repeater {
    static constexpr T value = (byte_repeater<T, count-1, byte>::value << CHAR_BIT) | byte;
};

template<typename T, int mask>
struct make_mask {
    using T2 = typename make_unsigned<T>::type;
    static constexpr T2 value = byte_repeater<T2, sizeof(T2), mask>::value;
};

template<typename T>
ostream& operator<<(ostream& os, const tuple<T, T>& obj);

template<typename T>
ostream& operator<<(ostream& os, const vector<T>& obj);

template<typename T1, typename T2>
void copyValues(const vector<T1>& src, vector<T2>& dst);

template<typename T>
ostream& operator<<(ostream& os, const vector<vector<T>>& obj);

#endif // UTILS_H
