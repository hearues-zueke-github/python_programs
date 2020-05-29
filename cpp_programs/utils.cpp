#include "utils.h"

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::tuple<T, T>& obj)
{
    os << "(";
    os << std::get<0>(obj) << ", " << std::get<1>(obj);
    os << ")";
    return os;
}
template std::ostream& operator<<(std::ostream& os, const std::tuple<int, int>& obj);

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& obj)
{
    // static constexpr auto width = sizeof(T) * 2;
    // static constexpr auto mask = make_mask<T,0xFF>::value;
    os << "[";
    std::for_each(obj.begin(), obj.end() - 1, [&os](const T& elem) {
        os << elem << ", ";
        // os << "0x" << std::hex << std::uppercase << std::setw(width) << std::setfill('0') << (elem & mask) << ", ";
    });
    os << obj.back();
    // os << "0x" << std::hex << std::uppercase << std::setw(width) << std::setfill('0') << (obj.back() & mask);
    os << "]";
    return os;
}

template<typename T1, typename T2>
void copyValues(const vector<T1>& src, vector<T2>& dst) {
  const size_t size = src.size();
  dst.resize(size);
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

template void copyValues(const vector<char>& src, vector<int16_t>& dst);
template void copyValues(const vector<char>& src, vector<int8_t>& dst);
// template void copyValues(const vector<int8_t>& src, vector<int16_t>& dst);

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
template ostream& operator<<(ostream& os, const std::vector<std::tuple<int, int>>& obj);
