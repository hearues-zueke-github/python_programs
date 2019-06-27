#include "utils.h"

#define GET_TEMPLATE_FUNCTION(NAMESPACE, WIDTH, TYPENAME, MASK) \
ostream& NAMESPACE::operator<<(ostream& os, const vector<TYPENAME>& obj) { \
  size_t size = obj.size(); \
  os << "["; \
  for (size_t i = 0; i < size; ++i) { \
    if (i > 0) { \
      os << ", "; \
    } \
    stringstream ss; \
    ss << "0x" << hex << uppercase << setw(WIDTH) << setfill('0') << ((+obj[i])&MASK); \
    os << ss.str(); \
  } \
  os << "]"; \
  return os; \
}

GET_TEMPLATE_FUNCTION(print_width_2, 2, int8_t, 0xFF)
GET_TEMPLATE_FUNCTION(print_width_2, 2, uint8_t, 0xFF)
using print_width_2::operator<<;

GET_TEMPLATE_FUNCTION(print_width_4, 4, int16_t, 0xFFFF)
GET_TEMPLATE_FUNCTION(print_width_4, 4, uint16_t, 0xFFFF)
using print_width_4::operator<<;

GET_TEMPLATE_FUNCTION(print_width_8, 8, int32_t, 0xFFFFFFFF)
GET_TEMPLATE_FUNCTION(print_width_8, 8, uint32_t, 0xFFFFFFFF)
using print_width_8::operator<<;

GET_TEMPLATE_FUNCTION(print_width_16, 16, int64_t, 0xFFFFFFFFFFFFFFFF)
GET_TEMPLATE_FUNCTION(print_width_16, 16, uint64_t, 0xFFFFFFFFFFFFFFFF)
using print_width_16::operator<<;

template<typename T>
ostream& operator<<(ostream& os, const vector<vector<T>>& obj) {
  const size_t size = obj.size();
  os << "[";
  for (size_t i = 0; i < size; ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << obj[i];
  }
  os << "]";
  return os;
}
template ostream& operator<<(ostream& os, const vector<vector<uint8_t>>& obj);
template ostream& operator<<(ostream& os, const vector<vector<int8_t>>& obj);
template ostream& operator<<(ostream& os, const vector<vector<uint16_t>>& obj);
template ostream& operator<<(ostream& os, const vector<vector<int16_t>>& obj);
template ostream& operator<<(ostream& os, const vector<vector<uint32_t>>& obj);
template ostream& operator<<(ostream& os, const vector<vector<int32_t>>& obj);
template ostream& operator<<(ostream& os, const vector<vector<uint64_t>>& obj);
template ostream& operator<<(ostream& os, const vector<vector<int64_t>>& obj);
