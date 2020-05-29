#include <algorithm>
#include <iostream>
#include <fstream>
#include <ostream>
#include <vector>
#include <stdint.h>

using std::cout;
using std::endl;

using std::vector;
using std::ostream;
using std::ofstream;
using std::ios;

ostream& operator<<(ostream& os, const uint8_t v) {
  os << +v;
  return os;
}

void write_vector_uint8_to_file(ofstream& f, const vector<vector<uint8_t>>& v) {
  const size_t s = v.size();
  for (size_t i = 0; i < s; ++i) {
    const vector<uint8_t>& row = v[i];
    f.write((char*)&(row[0]), row.size());
  }
}

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

template ostream& operator<<(ostream& os, const vector<vector<uint8_t>>& obj);
template ostream& operator<<(ostream& os, const vector<uint8_t>& obj);

int main(int argc, char* argv[]) {
  cout << "Hello World!" << endl;

  // const uint32_t n = 5;
  const uint32_t n = 12;
  
  auto calc_factorial = [](const uint64_t n) {
    uint64_t f = 1;
    for (uint64_t i = 1; i <= n; ++i) {
      f *= i;
    }
    return f;
  };

  size_t fac_size = calc_factorial(n);
  cout << "fac_size: " << fac_size << endl;
  // uint8_t* a{ new uint8_t[fac_size * n]{} };



  // vector<uint8_t> arr_one = new vector<uint8_t>(1);
  // vector<vector<uint8_t>> arr(1);

  // vector<uint8_t> arr_one = new vector<uint8_t>(fac_size * n);
  vector<vector<uint8_t>> arr;
  arr.resize(fac_size);
  for (size_t i = 0; i < fac_size; ++i) {
    arr[i].resize(n);
  }
  // vector<vector<uint8_t>> arr(fac_size, vector<uint8_t>(n));
  // vector<vector<uint8_t>> arr(fac_size);

  // delete[] a;

  // return 0;

  const size_t l1 = arr.size();
  const size_t l2 = arr[0].size();

  cout << "l1: " << l1 << endl;
  cout << "l2: " << l2 << endl;

  if (n >= 1) {
    vector<uint8_t>& row1 = arr[0];
    for (size_t i = 0; i < n; ++i) {
      row1[i] = i;
    }
  }
  if (n >= 2) {
    vector<uint8_t>& row = arr[1];
    row = arr[0];
    row[0] = 1;
    row[1] = 0;
  }

  // TODO: write the algorithm for creating the permutation table!
  size_t amount_rows_to_copy = 2;
  for (size_t j = 2; j < n; ++j) {
    for (size_t mult = 0; mult < j; ++mult) {
      const size_t mult_1 = mult + 1;
      for (size_t k = 0; k < amount_rows_to_copy; ++k) {
        vector<uint8_t>& row = arr[amount_rows_to_copy * mult_1 + k];
        row = arr[k];
        row[j] = mult;
        for (size_t l = 0; l < j; ++l) {
          if (row[l] == mult) {
            row[l] = j;
            break;
          }
        }
      }
    }
    amount_rows_to_copy *= (j+1);
  }

  // cout << "arr: " << arr << endl;

  ofstream myfile;
  myfile.open("test_permutation.hex", ios::binary);

  write_vector_uint8_to_file(myfile, arr);

  myfile.close();

  return 0;
}
