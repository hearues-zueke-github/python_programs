#include "utils_primes.h"

template<typename T>
ostream& operator<<(ostream& os, const std::vector<T>& obj) {
  os << "[";
  for_each(obj.begin(), obj.end() - 1, [&os](const T& elem) {
    os << elem << ", ";
  });
  os << obj.back();
  os << "]";
  return os;
}

// template ostream& operator<<(ostream& os, const vector<vector<uint8_t>>& obj);
template ostream& operator<<(ostream& os, const vector<uint8_t>& obj);
template ostream& operator<<(ostream& os, const vector<uint16_t>& obj);
template ostream& operator<<(ostream& os, const vector<uint32_t>& obj);
template ostream& operator<<(ostream& os, const vector<uint64_t>& obj);
template ostream& operator<<(ostream& os, const vector<vector<uint64_t>>& obj);

template<typename T>
inline void subVector(const vector<T>& src, vector<T>& dst, int m, int n) {
    auto first = src.begin() + m;
    auto last = src.begin() + n;
    dst.resize(0);
    dst.insert(dst.end(), first, last);
}

template void subVector(const vector<uint64_t>& src, vector<uint64_t>& dst, int m, int n);
