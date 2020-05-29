#include <iostream>
#include <vector>
#include <assert.h>
#include <sys/types.h>
#include <ostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>

using std::cout;
using std::endl;
using std::ostream;
using std::ios;
using std::ifstream;
using std::stringstream;
using std::hex;
using std::setw;
using std::setfill;

using std::vector;

using i64 = int64_t;
using u64 = uint64_t;
using i8 = int8_t;
using u8 = uint8_t;
using u16 = uint16_t;

namespace utils {
  auto setStringStream = [](stringstream& ss, int width) {
    ss << hex << std::uppercase << setw(width) << setfill('0');
  };
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& obj)
{
  os << "[";
  std::for_each(obj.begin(), obj.end() - 1, [&os](const T& elem) {
    os << elem << ", ";
  });
  os << obj.back();
  os << "]";
  return os;
}

template ostream& operator<<(ostream& os, const vector<vector<i64>>& obj);
template ostream& operator<<(ostream& os, const vector<i64>& obj);

inline void copyRow(vector<i64>::iterator src_begin, vector<i64>::iterator src_end, vector<i64>& dst) {
  size_t i = 0;
  for (vector<i64>::iterator it = src_begin, it2 = dst.begin(); it != src_end; i++, it++, it2++) {
    *it2 = *it;
  }
}

class RandomNumber {
private:
  size_t _index;
  vector<u64> _buffer_64;
  ifstream _urandom;
  u64 _seed;
  size_t _amount;
public:
  RandomNumber();
  u64 nextNum();
};

RandomNumber::RandomNumber() : _index(0), _seed(0), _amount(1024) {
  _urandom = ifstream("/dev/urandom", ios::in|ios::binary);
  _buffer_64.resize(_amount);
  _urandom.read(reinterpret_cast<char*> (&_buffer_64[0]), 8 * _amount);
}

u64 RandomNumber::nextNum() {
  if (_index >= _amount) {
    _index = 0;
    _urandom.read(reinterpret_cast<char*> (&_buffer_64[0]), 8 * _amount);
  }

  _index += 1;
  return _buffer_64[_index-1];
}

class Matrix {
private:
  size_t _y;
  size_t _x;
  i64 _default_val;
  vector<vector<i64>> _M;
  RandomNumber _rn;
public:
  Matrix(size_t y, size_t x, i64 default_val = 0);
  Matrix(vector<vector<i64>> M);
  Matrix(const Matrix &obj);
  ~Matrix() {};
  Matrix& operator=(const Matrix& other);
  friend Matrix operator+(const Matrix& A, const Matrix& B);
  friend Matrix operator+(const i64 a, const Matrix& M);
  friend Matrix operator+(const Matrix& M, const i64 a);
  friend Matrix operator-(const Matrix& A, const Matrix& B);
  friend Matrix operator-(const i64 a, const Matrix& M);
  friend Matrix operator-(const Matrix& M, const i64 a);
  friend Matrix operator*(const Matrix& A, const Matrix& B);
  friend Matrix operator*(const i64 a, const Matrix& M);
  friend Matrix operator*(const Matrix& M, const i64 a);
  friend ostream& operator<<(ostream& os, const Matrix& obj);
  friend bool operator==(const Matrix& lhs, const Matrix& rhs);
  friend bool operator!=(const Matrix& lhs, const Matrix& rhs);
  void fillRandom(i64 minVal, i64 maxVal);
  Matrix trans() const;
};

Matrix::Matrix(size_t y, size_t x, i64 default_val) : _y(y), _x(x), _default_val(default_val) {
  _M.resize(_y);
  for (auto it = _M.begin(); it != _M.end(); ++it) {
    vector<i64>& v = *it;
    v.resize(_x);
    std::fill(v.begin(), v.end(), _default_val);
  }
}

Matrix::Matrix(vector<vector<i64>> M) : _default_val(0) {
  // _default_val = 0;
  _y = M.size();
  assert(_y > 0);
  _x = M[0].size();
  assert(_x > 0);
  auto it_end = M.end();
  for (auto it = M.begin() + 1; it != it_end; ++it) {
    assert(_x == (*it).size());
  }

  _M = M;
}

Matrix::Matrix(const Matrix &obj) : _y(obj._y), _x(obj._x),
    _default_val(obj._default_val), _M(obj._M) {
}

Matrix& Matrix::operator=(const Matrix& other) {
  if (this != &other) { // self-assignment check expected
    this->_y = other._y;
    this->_x = other._x;
    this->_default_val = other._default_val;
    this->_M = other._M;
  }
  return *this;
}

Matrix operator+(const Matrix& A, const Matrix& B) {
  const size_t y = A._y;
  const size_t x = A._x;
  
  assert(y == B._y);
  assert(x == B._x);
  
  Matrix Y(y, x);
  
  vector<vector<i64>>& MY = Y._M;
  const vector<vector<i64>>& MA = A._M;
  const vector<vector<i64>>& MB = B._M;
  
  for (size_t j = 0; j < y; ++j) {
    vector<i64>& vY = MY[j];
    const vector<i64>& vA = MA[j];
    const vector<i64>& vB = MB[j];
    for (size_t i = 0; i < x; ++i) {
      vY[i] = vA[i] + vB[i];
    }
  }

  return Y;
}

Matrix operator+(const i64 a, const Matrix& M) {
  const size_t y = M._y;
  const size_t x = M._x;
  
  Matrix Y(y, x);
  
  vector<vector<i64>>& MY = Y._M;
  const vector<vector<i64>>& MM = M._M;
  
  for (size_t j = 0; j < y; ++j) {
    vector<i64>& vY = MY[j];
    const vector<i64>& vM = MM[j];
    for (size_t i = 0; i < x; ++i) {
      vY[i] = vM[i] + a;
    }
  }

  return Y;
}

Matrix operator+(const Matrix& M, const i64 a) {
  const size_t y = M._y;
  const size_t x = M._x;
  
  Matrix Y(y, x);
  
  vector<vector<i64>>& MY = Y._M;
  const vector<vector<i64>>& MM = M._M;
  
  for (size_t j = 0; j < y; ++j) {
    vector<i64>& vY = MY[j];
    const vector<i64>& vM = MM[j];
    for (size_t i = 0; i < x; ++i) {
      vY[i] = vM[i] + a;
    }
  }

  return Y;
}

Matrix operator-(const Matrix& A, const Matrix& B) {
  const size_t y = A._y;
  const size_t x = A._x;
  
  assert(y == B._y);
  assert(x == B._x);
  
  Matrix Y(y, x);
  
  vector<vector<i64>>& MY = Y._M;
  const vector<vector<i64>>& MA = A._M;
  const vector<vector<i64>>& MB = B._M;
  
  for (size_t j = 0; j < y; ++j) {
    vector<i64>& vY = MY[j];
    const vector<i64>& vA = MA[j];
    const vector<i64>& vB = MB[j];
    for (size_t i = 0; i < x; ++i) {
      vY[i] = vA[i] - vB[i];
    }
  }

  return Y;
}

Matrix operator-(const i64 a, const Matrix& M) {
  const size_t y = M._y;
  const size_t x = M._x;
  
  Matrix Y(y, x);
  
  vector<vector<i64>>& MY = Y._M;
  const vector<vector<i64>>& MM = M._M;
  
  for (size_t j = 0; j < y; ++j) {
    vector<i64>& vY = MY[j];
    const vector<i64>& vM = MM[j];
    for (size_t i = 0; i < x; ++i) {
      vY[i] = a - vM[i];
    }
  }

  return Y;
}

Matrix operator-(const Matrix& M, const i64 a) {
  const size_t y = M._y;
  const size_t x = M._x;
  
  Matrix Y(y, x);
  
  vector<vector<i64>>& MY = Y._M;
  const vector<vector<i64>>& MM = M._M;
  
  for (size_t j = 0; j < y; ++j) {
    vector<i64>& vY = MY[j];
    const vector<i64>& vM = MM[j];
    for (size_t i = 0; i < x; ++i) {
      vY[i] = vM[i] - a;
    }
  }

  return Y;
}

Matrix operator*(const Matrix& A, const Matrix& B) {
  const size_t yA = A._y;
  const size_t xA = A._x;
  const size_t yB = B._y;
  const size_t xB = B._x;
  
  assert(xA == yB);
  
  Matrix Y(yA, xB);
  
  const Matrix Bt = B.trans();

  vector<vector<i64>>& MY = Y._M;
  const vector<vector<i64>>& MA = A._M;
  const vector<vector<i64>>& MBt = Bt._M;
  
  for (size_t j = 0; j < yA; ++j) {
    vector<i64>& vY = MY[j];
    const vector<i64>& vA = MA[j];
    for (size_t i = 0; i < xB; ++i) {
      const vector<i64>& vBt = MBt[i];
      i64 s = 0;
      for (size_t k = 0; k < xA; ++k) {
        s += vA[k] * vBt[k];
      }
      vY[i] = s;
    }
  }

  return Y;
}

Matrix operator*(const i64 a, const Matrix& M) {
  const size_t y = M._y;
  const size_t x = M._x;
  
  Matrix Y(y, x);
  
  vector<vector<i64>>& MY = Y._M;
  const vector<vector<i64>>& MM = M._M;
  
  for (size_t j = 0; j < y; ++j) {
    vector<i64>& vY = MY[j];
    const vector<i64>& vM = MM[j];
    for (size_t i = 0; i < x; ++i) {
      vY[i] = vM[i] * a;
    }
  }

  return Y;
}

Matrix operator*(const Matrix& M, const i64 a) {
  const size_t y = M._y;
  const size_t x = M._x;
  
  Matrix Y(y, x);
  
  vector<vector<i64>>& MY = Y._M;
  const vector<vector<i64>>& MM = M._M;
  
  for (size_t j = 0; j < y; ++j) {
    vector<i64>& vY = MY[j];
    const vector<i64>& vM = MM[j];
    for (size_t i = 0; i < x; ++i) {
      vY[i] = vM[i] * a;
    }
  }

  return Y;
}

ostream& operator<<(ostream& os, const Matrix& obj) {
  os << obj._M;
  return os;
}

bool operator==(const Matrix& lhs, const Matrix& rhs) {
  if (lhs._y != rhs._y && lhs._x != rhs._x) {
    return false;
  }

  size_t y = lhs._y;
  size_t x = lhs._x;

  const vector<vector<i64>>& lM = lhs._M;
  const vector<vector<i64>>& rM = rhs._M;

  for (size_t j = 0; j < y; ++j) {
    const vector<i64>& lv = lM[j];
    const vector<i64>& rv = rM[j];
    for (size_t i = 0; i < x; ++i) {
      if (lv[i] != rv[i]) {
        return false;
      }
    }
  }

  return true;
}

bool operator!=(const Matrix& lhs, const Matrix& rhs) {
  return !(lhs == rhs);
}



void Matrix::fillRandom(i64 minVal, i64 maxVal) {
  u64 diff = (u64)(maxVal - minVal + 1);
  for (size_t j = 0; j < _y; ++j) {
    vector<i64>& v = _M[j];
    for (size_t i = 0; i < _x; ++i) {
      v[i] = minVal + (_rn.nextNum() % diff);
    }
  }
}

Matrix Matrix::trans() const {
  Matrix Y(_x, _y);

  vector<vector<i64>>& M = Y._M;

  for (size_t j = 0; j < _y; j++) {
    const vector<i64>& v = _M[j];
    for (size_t i = 0; i < _x; i++) {
      M[i][j] = v[i];
    }
  }

  return Y;
}


i64 detMatInt(vector<vector<i64>>& A) {
  size_t size = A.size();
  assert(size > 0);
  assert(size == A[0].size()); 

  if (size == 1) {
    return A[0][0];
  } else if (size == 2) {
    vector<i64> r1 = A[0];
    vector<i64> r2 = A[1];
    return r1[0]*r2[1]-r1[1]*r2[0];
  }

  i64 s = 0;

  vector<vector<i64>> A_smaller(size-1, vector<i64>(size-1, 0));
  // fill the upper right part of the orig A matrix into A_smaller
  for (size_t i = 0; i < size-1; i++) {
    vector<i64>& v = A[i];
    copyRow(v.begin()+1, v.end(), A_smaller[i]);
  }

  i64 sign = ((size % 2) == 1) ? 1 : -1;
  s += sign * A[size-1][0] * detMatInt(A_smaller);

  // create the smaller matrix here!
  for (i64 i = size - 2; i > -1; i--) {
    vector<i64>& v = A[i+1];
    copyRow(v.begin()+1, v.end(), A_smaller[i]);
    sign *= -1;
    i64 v0 = A[i][0];
    if (v0 == 0) {
      continue;
    }
    s += sign * v0 * detMatInt(A_smaller);
  }

  return s;
}

void test_own_determination() {
  vector<vector<i64>> A_1 = {
      {-9, -6,  6, -1},
      { 5,  4, -5, -4},
      { 0, -6, -7,  6},
      { 4, 10, -2,  7}
    };
    i64 det_A_1 = 3490;
    i64 det_A_1_calc = detMatInt(A_1);
    assert(det_A_1 == det_A_1_calc);

    vector<vector<i64>> A_2 = {
        { -2, 6, 5, -10, 6, -9},
        { 2, 2, 6, -9, -2, 9},
        { 3, 9, 1, -7, 2, 10},
        { 0, 5, -1, -2, -9, -9},
        {-10, 6, 9, -10, -3, 1},
        { 5, 1, -10, -4, -9, 4}
    };
    i64 det_A_2 = 2552976;
    i64 det_A_2_calc = detMatInt(A_2);
    assert(det_A_2 == det_A_2_calc);
}

int main(int argc, char* argv[]) {
  test_own_determination();

  RandomNumber rn = RandomNumber();

  auto create01Matrix = [](size_t n, vector<vector<i64>>& A, RandomNumber& rn) {
    A.resize(n);
    for (auto it = A.begin(); it != A.end(); ++it) {
      vector<i64>& v = *it;
      v.resize(n);
      std::fill(v.begin(), v.end(), 0);
    }

    for (size_t j = 0; j < n; ++j) {
      vector<i64>& v = A[j];
      for (size_t i = 0; i < n; ++i) {
        v[i] = rn.nextNum() % 2;
        // v[i] = rn.nextNum() % 11 - 5;
      }
    }
  };

  vector<vector<i64>> A;

  i64 max_det_A = 0;
  vector<vector<i64>> max_A;
  size_t n = 5;
  for (int j = 0; j < 10000; ++j) {
    if (j % 1000 == 0) {
      cout << "j: " << j << endl;
    }
    create01Matrix(n, A, rn);
    i64 det_A = detMatInt(A);
    
    if (max_det_A < det_A) {
      max_det_A = det_A;
      max_A = A;
    }
  }

  cout << endl;
  cout << "n: " << n << endl;
  cout << "max_det_A: " << max_det_A << endl;
  cout << "max_A: " << max_A << endl;

  // TODO: create test cases for the Matrix class

  // Matrix A1 = Matrix(3, 4);
  Matrix A1 = Matrix({{1, 2, 3, 4}, {2, 3, 4, 5}, {4, 5, 6, 7}});
  // A1.fillRandom(3, 5);
  cout << "A1: " << A1 << endl;
  // cout << "random fill: A1: " << A1 << endl;

  Matrix A2 = Matrix(3, 4);
  A2.fillRandom(-3, 3);
  cout << "A2: " << A2 << endl;
  // cout << "random fill: A2: " << A2 << endl;

  Matrix A3 = A1 + A2;
  cout << "A3 = A1 + A2: " << A3 << endl;

  cout << "A1+5: " << (A1+5) << endl;

  Matrix A4 = A1.trans();
  cout << "A4: " << A4 << endl;

  

  // test matrix dot operation
  Matrix A11 = Matrix({{1, 2}, {3, 4}, {4, 5}});
  Matrix A12 = Matrix({{1, 2, 4}, {3, 4, -2}});

  Matrix A11_12 = A11 * A12;
  Matrix A11_12_ref = Matrix({{7, 10, 0}, {15, 22, 4}, {19, 28, 6}});
  assert(A11_12 == A11_12_ref);

  return 0;
}
