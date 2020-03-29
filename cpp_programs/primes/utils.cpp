#include <vector>
#include <stdint.h>
#include "utils_primes.h"

template<typename T>
ostream& operator<<(ostream& os, const std::vector<T>& obj)
{
  os << "[";
  std::for_each(obj.begin(), obj.end() - 1, [&os](const T& elem) {
    os << elem << ", ";
  });
  os << obj.back();
  os << "]";
  return os;
}

// template ostream& operator<<(ostream& os, const vector<vector<uint8_t>>& obj);
template ostream& operator<<(ostream& os, const vector<uint8_t>& obj);
