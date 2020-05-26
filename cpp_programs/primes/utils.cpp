#include "utils.h"

ostream& operator<<(ostream& os, const std::vector<uint8_t>& obj) {
  os << "[";
  std::for_each(obj.begin(), obj.end() - 1, [&os](const uint8_t elem) {
    os << unsigned(elem) << ", ";
  });
  os << unsigned(obj.back());
  os << "]";
  return os;
}

template<typename T>
ostream& operator<<(ostream& os, const std::vector<T>& obj) {
  os << "[";
  std::for_each(obj.begin(), obj.end() - 1, [&os](const T& elem) {
    os << elem << ", ";
  });
  os << obj.back();
  os << "]";
  return os;
}

template ostream& operator<<(ostream& os, const vector<uint16_t>& obj);
template ostream& operator<<(ostream& os, const vector<vector<uint8_t>>& obj);
