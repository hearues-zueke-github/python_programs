//
// Created by doublepmcl on 14.06.19.
//

#ifndef SEQUENCESGENERATOR_NUMBER_H
#define SEQUENCESGENERATOR_NUMBER_H

#include <cstdint>
#include <cmath>

using namespace std;

template<typename T>
class Number;

template<typename T> Number<T> operator+(const Number<T>& lhs, const Number<T>& rhs);
template<typename T> Number<T> operator%(const Number<T>& lhs, const Number<T>& rhs);

template<typename T>
class Number {
private:
  T _n;
public:
  Number(T n);
  Number(const Number<T>& obj);
  virtual ~Number();

  friend Number<T> operator+ <>(const Number<T>& lhs, const Number<T>& rhs);
  friend Number<T> operator% <>(const Number<T>& lhs, const Number<T>& rhs);

  friend Number<float> operator%(const Number<float>& lhs, const Number<float>& rhs);
  friend Number<double> operator%(const Number<double>& lhs, const Number<double>& rhs);
  friend Number<long double> operator%(const Number<long double>& lhs, const Number<long double>& rhs);
};


#endif //SEQUENCESGENERATOR_NUMBER_H
