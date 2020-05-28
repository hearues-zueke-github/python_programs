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

ostream& operator<<(ostream& os, const std::vector<string>& obj) {
  os << "[";
  std::for_each(obj.begin(), obj.end() - 1, [&os](const string elem) {
    os << "\"" << elem << "\"" << ", ";
  });
  os << "\"" << obj.back() << "\"";
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

// template ostream& operator<<(ostream& os, const vector<vector<uint8_t>>& obj);

// template ostream& operator<<(ostream& os, const vector<string>& obj);

template ostream& operator<<(ostream& os, const vector<uint16_t>& obj);
template ostream& operator<<(ostream& os, const vector<int>& obj);
template ostream& operator<<(ostream& os, const vector<vector<int>>& obj);
