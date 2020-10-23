//
// Created by doublepmcl on 06.06.19.
//

#include "Numbers.h"

Numbers::Numbers() : _nums(vector<uint64_t>()) {
}

Numbers::Numbers(const vector<uint64_t>& nums) : _nums(nums) {
}

Numbers::Numbers(const Numbers& obj) : _nums(obj._nums) {
}

Numbers::~Numbers() {
  _nums.clear();
}

vector<uint64_t>& Numbers::getNums() {
  return _nums;
}

ostream& operator<<(ostream& os, const Numbers& obj) {
  os << "[";
  vector<uint64_t> v = obj._nums;
  size_t l = v.size();
  for (size_t i = 0; i < l; ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << v[i];
  }
  os << "]";
  return os;
}

Numbers::Numbers() : _nums(vector<uint64_t>()) {
}

Numbers::Numbers(const vector<uint64_t>& nums) : _nums(nums) {
}

Numbers::Numbers(const Numbers& obj) : _nums(obj._nums) {
}

Numbers::~Numbers() {
  _nums.clear();
}

vector<uint64_t>& Numbers::getNums() {
  return _nums;
}

ostream& operator<<(ostream& os, const Numbers& obj) {
  os << "[";
  vector<uint64_t> v = obj._nums;
  size_t l = v.size();
  for (size_t i = 0; i < l; ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << v[i];
  }
  os << "]";
  return os;
}
