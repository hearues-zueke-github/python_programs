//
// Created by doublepmcl on 13.06.19.
//

#ifndef SEQUENCESGENERATOR_MATRIX_H
#define SEQUENCESGENERATOR_MATRIX_H

#include <cmath>
#include <exception>
#include <ostream>
#include <string>
#include <vector>

using namespace std;

template<typename T>
class Matrix;

template<typename T> ostream& operator<<(ostream& os, Matrix<T> const & obj);
template<typename T> Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2);
template<typename T> Matrix<T> operator-(const Matrix<T>& m1, const Matrix<T>& m2);
template<typename T> Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2);
template<typename T> Matrix<T> operator/(const Matrix<T>& m1, const Matrix<T>& m2);
template<typename T> Matrix<T> operator%(const Matrix<T>& m1, const Matrix<T>& m2);

//Matrix<float> mod(const float& v);
//Matrix<double> mod(const double& v);
//Matrix<long double> mod(const long double& v);

template<typename T>
class Matrix {
private:
  int _y;
  int _x;
  vector<vector<T>> _A;
public:
  Matrix(const int y, const int x);
  Matrix(const Matrix<T>& obj);
  virtual ~Matrix();

  T& operator()(const int y, const int x);
  T operator()(const int y, const int x) const;

  friend ostream& operator<< <>(ostream& os, Matrix<T> const & obj);

  Matrix<T> dot(const Matrix<T>& o);
  Matrix<T> mod(const T& v);

  friend Matrix<T> operator+ <>(const Matrix<T>& m1, const Matrix<T>& m2);
  friend Matrix<T> operator- <>(const Matrix<T>& m1, const Matrix<T>& m2);
  friend Matrix<T> operator* <>(const Matrix<T>& m1, const Matrix<T>& m2);
  friend Matrix<T> operator/ <>(const Matrix<T>& m1, const Matrix<T>& m2);
  friend Matrix<T> operator% <>(const Matrix<T>& m1, const Matrix<T>& m2);

  friend Matrix<float> operator%(const Matrix<float>& m1, const Matrix<float>& m2);
  friend Matrix<double> operator%(const Matrix<double>& m1, const Matrix<double>& m2);
  friend Matrix<long double> operator%(const Matrix<long double>& m1, const Matrix<long double>& m2);
};

#endif //SEQUENCESGENERATOR_MATRIX_H
