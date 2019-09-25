#ifndef MODULO_SEQUENCE_UTILS_H
#define MODULO_SEQUENCE_UTILS_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <sstream>

using namespace std;

namespace print_width_2 {
  ostream& operator<<(ostream& os, const vector<int8_t>& obj);
  ostream& operator<<(ostream& os, const vector<uint8_t>& obj);
}

namespace print_width_4 {
  ostream& operator<<(ostream& os, const vector<int16_t>& obj);
  ostream& operator<<(ostream& os, const vector<uint16_t>& obj);
}

namespace print_width_8 {
  ostream& operator<<(ostream& os, const vector<int32_t>& obj);
  ostream& operator<<(ostream& os, const vector<uint32_t>& obj);
}

namespace print_width_16 {
  ostream& operator<<(ostream& os, const vector<int64_t>& obj);
  ostream& operator<<(ostream& os, const vector<uint64_t>& obj);
}

template<typename T>
ostream& operator<<(ostream& os, const vector<vector<T>>& obj);

using print_width_2::operator<<;
using print_width_4::operator<<;
using print_width_8::operator<<;
using print_width_16::operator<<;

#endif // MODULO_SEQUENCE_UTILS_H
