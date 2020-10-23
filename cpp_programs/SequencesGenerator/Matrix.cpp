//
// Created by doublepmcl on 13.06.19.
//

#include "Matrix.h"

struct DivisionByZero : public exception {
  int _y;
  int _x;
  string _msg;
  DivisionByZero(int y, int x) : _y(y), _x(x), _msg() {
    char temp[50];
    sprintf(temp, "Found 0 at the position: (y, x) = (%d, %d)", _y, _x);
    _msg = temp;
  }
  const char* what() const throw () {
    return _msg.c_str();
  }
};

struct ModuloSmaller1 : public exception {
  int _y;
  int _x;
  string _msg;
  ModuloSmaller1(int y, int x) : _y(y), _x(x), _msg() {
    char temp[50];
    sprintf(temp, "Found 0 at the position: (y, x) = (%d, %d)", _y, _x);
    _msg = temp;
  }
  const char* what() const throw () {
    return _msg.c_str();
  }
};

struct ModuloByZero : public exception {
  int _y;
  int _x;
  string _msg;
  ModuloByZero(int y, int x) : _y(y), _x(x), _msg() {
    char temp[50];
    sprintf(temp, "Found 0 at the position: (y, x) = (%d, %d)", _y, _x);
    _msg = temp;
  }
  const char* what() const throw () {
    return _msg.c_str();
  }
};

template<typename T>
Matrix<T>::Matrix(const int y, const int x) :
    _y(y),_x(x),
    _A(vector<vector<T>>(y, vector<T>(x, 0))) {
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& obj) :
    _y(obj._y),
    _x(obj._x),
    _A(obj._A) {
}

template<typename T>
Matrix<T>::~Matrix() {
  _A.clear();
}

template<typename T>
T& Matrix<T>::operator()(const int y, const int x) {
  return _A[y][x];
}

template<typename T>
T Matrix<T>::operator()(const int y, const int x) const {
  return _A[y][x];
}

template<typename T>
ostream& operator<<(ostream& os, Matrix<T> const & obj) {
  os << "[";
  for (size_t y = 0; y < obj._y; ++y) {
    if (y > 0) {
      os << ", [";
    } else {
      os << "[";
    }
    for (size_t x = 0; x < obj._x; ++x) {
      if (x > 0) {
        os << ", ";
      }
      os << obj._A[y][x];
    }
    os << "]";
  }
  os << "]";
  return os;
}

template<typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& o) {
  Matrix<T> m(_y, o._x);
  // TODO: exception if _x != o._y !!!

  for (size_t y = 0; y < m._y; ++y) {
    for (size_t x = 0; x < m._x; ++x) {
      T s = 0;
      for (size_t k = 0; k < _x; ++k) {
        s += _A[y][k] * o._A[k][x];
      }
      m._A[y][x] = s;
    }
  }

  return m;
}

template<typename T>
Matrix<T> Matrix<T>::mod(const T& v) {
  Matrix<T> m(*this);

  for (size_t y = 0; y < _y; ++y) {
    for (size_t x = 0; x < _x; ++x) {
      m._A[y][x] %= v;
    }
  }

  return m;
}

template<>
Matrix<float> Matrix<float>::mod(const float& v) {
  Matrix<float> m(*this);

  for (size_t y = 0; y < _y; ++y) {
    for (size_t x = 0; x < _x; ++x) {
      m._A[y][x] = fmod(m._A[y][x], v);
    }
  }

  return m;
}

template<>
Matrix<double> Matrix<double>::mod(const double& v) {
  Matrix<double> m(*this);

  for (size_t y = 0; y < _y; ++y) {
    for (size_t x = 0; x < _x; ++x) {
      m._A[y][x] = fmod(m._A[y][x], v);
    }
  }

  return m;
}

template<>
Matrix<long double> Matrix<long double>::mod(const long double& v) {
  Matrix<long double> m(*this);

  for (size_t y = 0; y < _y; ++y) {
    for (size_t x = 0; x < _x; ++x) {
      m._A[y][x] = fmod(m._A[y][x], v);
    }
  }

  return m;
}

template<typename T>
Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2) {
  Matrix<T> m3(m1);
  for (size_t y = 0; y < m1._y; ++y) {
    for (size_t x = 0; x < m1._x; ++x) {
      m3._A[y][x] += m2._A[y][x];
    }
  }
  return m3;
}

template<typename T>
Matrix<T> operator-(const Matrix<T>& m1, const Matrix<T>& m2) {
  Matrix<T> m3(m1);
  for (size_t y = 0; y < m1._y; ++y) {
    for (size_t x = 0; x < m1._x; ++x) {
      m3._A[y][x] -= m2._A[y][x];
    }
  }
  return m3;
}

template<typename T>
Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2) {
  Matrix<T> m3(m1);
  for (size_t y = 0; y < m1._y; ++y) {
    for (size_t x = 0; x < m1._x; ++x) {
      m3._A[y][x] *= m2._A[y][x];
    }
  }
  return m3;
}

template<typename T>
Matrix<T> operator/(const Matrix<T>& m1, const Matrix<T>& m2) {
  Matrix<T> m3(m1);
  for (size_t y = 0; y < m1._y; ++y) {
    for (size_t x = 0; x < m1._x; ++x) {
      T v = m2._A[y][x];
      if (v == 0) {
        throw DivisionByZero(y, x);
      }
      m3._A[y][x] /= v;

    }
  }
  return m3;
}

template<typename T>
Matrix<T> operator%(const Matrix<T>& m1, const Matrix<T>& m2) {
  Matrix<T> m3(m1);
  for (size_t y = 0; y < m1._y; ++y) {
    for (size_t x = 0; x < m1._x; ++x) {
      T v = m2._A[y][x];
      if (v < 1) {
        throw ModuloSmaller1(y, x);
      }
      m3._A[y][x] %= v;
    }
  }
  return m3;
}

template class Matrix<uint8_t>;
template class Matrix<uint16_t>;
template class Matrix<uint32_t>;
template class Matrix<uint64_t>;
template class Matrix<int8_t>;
template class Matrix<int16_t>;
template class Matrix<int32_t>;
template class Matrix<int64_t>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<long double>;

template ostream& operator<<(ostream& os, const Matrix<uint8_t>& obj);
template ostream& operator<<(ostream& os, const Matrix<uint16_t>& obj);
template ostream& operator<<(ostream& os, const Matrix<uint32_t>& obj);
template ostream& operator<<(ostream& os, const Matrix<uint64_t>& obj);
template ostream& operator<<(ostream& os, const Matrix<int8_t>& obj);
template ostream& operator<<(ostream& os, const Matrix<int16_t>& obj);
template ostream& operator<<(ostream& os, const Matrix<int32_t>& obj);
template ostream& operator<<(ostream& os, const Matrix<int64_t>& obj);
template ostream& operator<<(ostream& os, const Matrix<float>& obj);
template ostream& operator<<(ostream& os, const Matrix<double>& obj);
template ostream& operator<<(ostream& os, const Matrix<long double>& obj);

template Matrix<uint8_t> operator+(const Matrix<uint8_t>& m1, const Matrix<uint8_t>& m2);
template Matrix<uint16_t> operator+(const Matrix<uint16_t>& m1, const Matrix<uint16_t>& m2);
template Matrix<uint32_t> operator+(const Matrix<uint32_t>& m1, const Matrix<uint32_t>& m2);
template Matrix<uint64_t> operator+(const Matrix<uint64_t>& m1, const Matrix<uint64_t>& m2);
template Matrix<int8_t> operator+(const Matrix<int8_t>& m1, const Matrix<int8_t>& m2);
template Matrix<int16_t> operator+(const Matrix<int16_t>& m1, const Matrix<int16_t>& m2);
template Matrix<int32_t> operator+(const Matrix<int32_t>& m1, const Matrix<int32_t>& m2);
template Matrix<int64_t> operator+(const Matrix<int64_t>& m1, const Matrix<int64_t>& m2);
template Matrix<float> operator+(const Matrix<float>& m1, const Matrix<float>& m2);
template Matrix<double> operator+(const Matrix<double>& m1, const Matrix<double>& m2);
template Matrix<long double> operator+(const Matrix<long double>& m1, const Matrix<long double>& m2);

template Matrix<uint8_t> operator-(const Matrix<uint8_t>& m1, const Matrix<uint8_t>& m2);
template Matrix<uint16_t> operator-(const Matrix<uint16_t>& m1, const Matrix<uint16_t>& m2);
template Matrix<uint32_t> operator-(const Matrix<uint32_t>& m1, const Matrix<uint32_t>& m2);
template Matrix<uint64_t> operator-(const Matrix<uint64_t>& m1, const Matrix<uint64_t>& m2);
template Matrix<int8_t> operator-(const Matrix<int8_t>& m1, const Matrix<int8_t>& m2);
template Matrix<int16_t> operator-(const Matrix<int16_t>& m1, const Matrix<int16_t>& m2);
template Matrix<int32_t> operator-(const Matrix<int32_t>& m1, const Matrix<int32_t>& m2);
template Matrix<int64_t> operator-(const Matrix<int64_t>& m1, const Matrix<int64_t>& m2);
template Matrix<float> operator-(const Matrix<float>& m1, const Matrix<float>& m2);
template Matrix<double> operator-(const Matrix<double>& m1, const Matrix<double>& m2);
template Matrix<long double> operator-(const Matrix<long double>& m1, const Matrix<long double>& m2);

template Matrix<uint8_t> operator*(const Matrix<uint8_t>& m1, const Matrix<uint8_t>& m2);
template Matrix<uint16_t> operator*(const Matrix<uint16_t>& m1, const Matrix<uint16_t>& m2);
template Matrix<uint32_t> operator*(const Matrix<uint32_t>& m1, const Matrix<uint32_t>& m2);
template Matrix<uint64_t> operator*(const Matrix<uint64_t>& m1, const Matrix<uint64_t>& m2);
template Matrix<int8_t> operator*(const Matrix<int8_t>& m1, const Matrix<int8_t>& m2);
template Matrix<int16_t> operator*(const Matrix<int16_t>& m1, const Matrix<int16_t>& m2);
template Matrix<int32_t> operator*(const Matrix<int32_t>& m1, const Matrix<int32_t>& m2);
template Matrix<int64_t> operator*(const Matrix<int64_t>& m1, const Matrix<int64_t>& m2);
template Matrix<float> operator*(const Matrix<float>& m1, const Matrix<float>& m2);
template Matrix<double> operator*(const Matrix<double>& m1, const Matrix<double>& m2);
template Matrix<long double> operator*(const Matrix<long double>& m1, const Matrix<long double>& m2);

template Matrix<uint8_t> operator/(const Matrix<uint8_t>& m1, const Matrix<uint8_t>& m2);
template Matrix<uint16_t> operator/(const Matrix<uint16_t>& m1, const Matrix<uint16_t>& m2);
template Matrix<uint32_t> operator/(const Matrix<uint32_t>& m1, const Matrix<uint32_t>& m2);
template Matrix<uint64_t> operator/(const Matrix<uint64_t>& m1, const Matrix<uint64_t>& m2);
template Matrix<int8_t> operator/(const Matrix<int8_t>& m1, const Matrix<int8_t>& m2);
template Matrix<int16_t> operator/(const Matrix<int16_t>& m1, const Matrix<int16_t>& m2);
template Matrix<int32_t> operator/(const Matrix<int32_t>& m1, const Matrix<int32_t>& m2);
template Matrix<int64_t> operator/(const Matrix<int64_t>& m1, const Matrix<int64_t>& m2);
template Matrix<float> operator/(const Matrix<float>& m1, const Matrix<float>& m2);
template Matrix<double> operator/(const Matrix<double>& m1, const Matrix<double>& m2);
template Matrix<long double> operator/(const Matrix<long double>& m1, const Matrix<long double>& m2);

template Matrix<uint8_t> operator%(const Matrix<uint8_t>& m1, const Matrix<uint8_t>& m2);
template Matrix<uint16_t> operator%(const Matrix<uint16_t>& m1, const Matrix<uint16_t>& m2);
template Matrix<uint32_t> operator%(const Matrix<uint32_t>& m1, const Matrix<uint32_t>& m2);
template Matrix<uint64_t> operator%(const Matrix<uint64_t>& m1, const Matrix<uint64_t>& m2);
template Matrix<int8_t> operator%(const Matrix<int8_t>& m1, const Matrix<int8_t>& m2);
template Matrix<int16_t> operator%(const Matrix<int16_t>& m1, const Matrix<int16_t>& m2);
template Matrix<int32_t> operator%(const Matrix<int32_t>& m1, const Matrix<int32_t>& m2);
template Matrix<int64_t> operator%(const Matrix<int64_t>& m1, const Matrix<int64_t>& m2);
Matrix<float> operator%(const Matrix<float>& m1, const Matrix<float>& m2) {
  Matrix<float> m3(m1);
  for (size_t y = 0; y < m1._y; ++y) {
    for (size_t x = 0; x < m1._x; ++x) {
      float v = m2._A[y][x];
      if (v < 1) {
        throw ModuloByZero(y, x);
      }
      m3._A[y][x] = fmod(m3._A[y][x], v);
    }
  }
  return m3;
}
Matrix<double> operator%(const Matrix<double>& m1, const Matrix<double>& m2){
  Matrix<double> m3(m1);
  for (size_t y = 0; y < m1._y; ++y) {
    for (size_t x = 0; x < m1._x; ++x) {
      double v = m2._A[y][x];
      if (v < 1) {
        throw ModuloByZero(y, x);
      }
      m3._A[y][x] = fmod(m3._A[y][x], v);
    }
  }
  return m3;
}
Matrix<long double> operator%(const Matrix<long double>& m1, const Matrix<long double>& m2){
  Matrix<long double> m3(m1);
  for (size_t y = 0; y < m1._y; ++y) {
    for (size_t x = 0; x < m1._x; ++x) {
      long double v = m2._A[y][x];
      if (v < 1) {
        throw ModuloByZero(y, x);
      }
      m3._A[y][x] = fmod(m3._A[y][x], v);
    }
  }
  return m3;
}
